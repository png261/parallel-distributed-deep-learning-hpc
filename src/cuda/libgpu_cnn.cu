/**
 * Native CUDA C++ Bridge for sm_61 (GTX 1080 Ti & GTX 1080) on HPC
 * Implements:
 * 1. Single GPU CNN Forward+Backward
 * 2. Data Parallelism (2x GPUs: GTX 1080 Ti + GTX 1080)
 * 3. Model Parallelism (Pipeline Parallelism across Dual GPUs):
 *    - GPU 0 (GTX 1080 Ti): Stage 1 (Conv1 Layer: im2col + GEMM)
 *    - Host-staged DMA Transfer across PCIe between GPU 0 and GPU 1
 *    - GPU 1 (GTX 1080): Stage 2 (Conv2 Layer + FC Layer)
 *    - Backward Pass: GPU 1 backprop -> Grad transfer GPU 1 to GPU 0 -> GPU 0 backprop
 */

#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>

#define TILE_SIZE 16

__global__ void im2col_batched_kernel(const float* __restrict__ data_im,
                                      int batch_size, int channels,
                                      int height, int width,
                                      int ksize, int stride,
                                      int height_col, int width_col,
                                      float* __restrict__ data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int patches_per_img = height_col * width_col;
    int patch_len = channels * ksize * ksize;
    int total_elements = batch_size * patches_per_img * patch_len;

    if (index < total_elements) {
        int row_idx = index / patch_len;
        int col_idx = index % patch_len;

        int b = row_idx / patches_per_img;
        int p = row_idx % patches_per_img;
        int h_col = p / width_col;
        int w_col = p % width_col;

        int c = col_idx / (ksize * ksize);
        int rem = col_idx % (ksize * ksize);
        int kh = rem / ksize;
        int kw = rem % ksize;

        int h_im = h_col * stride + kh;
        int w_im = w_col * stride + kw;

        data_col[index] = data_im[((b * channels + c) * height + h_im) * width + w_im];
    }
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" {

// Data Parallelism Benchmark (1x GPU or 2x GPUs)
double benchmark_cnn_step_multigpu(int batch_size, int num_gpus, int num_runs) {
    int local_bs = batch_size / num_gpus;
    int img_h = 224, img_w = 224, ksize = 11, stride = 4, num_filters = 96;
    int h_col = (img_h - ksize) / stride + 1;
    int w_col = (img_w - ksize) / stride + 1;
    int M = local_bs * h_col * w_col;
    int K = ksize * ksize;
    int N = num_filters;

    std::vector<float*> d_im(num_gpus), d_col(num_gpus), d_w(num_gpus), d_out(num_gpus), d_grad_w(num_gpus);

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaMalloc(&d_im[g], local_bs * img_h * img_w * sizeof(float));
        cudaMalloc(&d_col[g], M * K * sizeof(float));
        cudaMalloc(&d_w[g], K * N * sizeof(float));
        cudaMalloc(&d_out[g], M * N * sizeof(float));
        cudaMalloc(&d_grad_w[g], K * N * sizeof(float));
    }

    int threads = 256;
    int blocks_im2col = (M * K + threads - 1) / threads;
    dim3 block_gemm(TILE_SIZE, TILE_SIZE);
    dim3 grid_fwd((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_bwd_w((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_bwd_x((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    #pragma omp parallel for num_threads(num_gpus)
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        im2col_batched_kernel<<<blocks_im2col, threads>>>(d_im[g], local_bs, 1, img_h, img_w, ksize, stride, h_col, w_col, d_col[g]);
        gemm_tiled_kernel<<<grid_fwd, block_gemm>>>(d_col[g], d_w[g], d_out[g], M, K, N);
        gemm_tiled_kernel<<<grid_bwd_w, block_gemm>>>(d_col[g], d_out[g], d_grad_w[g], K, M, N);
        gemm_tiled_kernel<<<grid_bwd_x, block_gemm>>>(d_out[g], d_w[g], d_col[g], M, N, K);
        cudaDeviceSynchronize();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < num_runs; ++r) {
        #pragma omp parallel for num_threads(num_gpus)
        for (int g = 0; g < num_gpus; ++g) {
            cudaSetDevice(g);
            im2col_batched_kernel<<<blocks_im2col, threads>>>(d_im[g], local_bs, 1, img_h, img_w, ksize, stride, h_col, w_col, d_col[g]);
            gemm_tiled_kernel<<<grid_fwd, block_gemm>>>(d_col[g], d_w[g], d_out[g], M, K, N);
            gemm_tiled_kernel<<<grid_bwd_w, block_gemm>>>(d_col[g], d_out[g], d_grad_w[g], K, M, N);
            gemm_tiled_kernel<<<grid_bwd_x, block_gemm>>>(d_out[g], d_w[g], d_col[g], M, N, K);
            cudaDeviceSynchronize();
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_runs;

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaFree(d_im[g]);
        cudaFree(d_col[g]);
        cudaFree(d_w[g]);
        cudaFree(d_out[g]);
        cudaFree(d_grad_w[g]);
    }
    return avg_ms;
}

// Heterogeneous Multi-GPU Benchmark across GPU 0 (GTX 1080 Ti) and GPU 1 (GTX 1080)
// Takes batch_0 (for GPU 0) and batch_1 (for GPU 1), runs concurrent Fwd+Bwd passes,
// and measures individual GPU execution times, total step time, and GPU 0 idle time.
void benchmark_heterogeneous_multigpu(int batch_0, int batch_1, int num_runs,
                                      double* out_gpu0_ms, double* out_gpu1_ms,
                                      double* out_total_ms, double* out_gpu0_idle_ms) {
    int img_h = 224, img_w = 224, ksize = 11, stride = 4, num_filters = 96;
    int h_col = (img_h - ksize) / stride + 1;
    int w_col = (img_w - ksize) / stride + 1;
    int K = ksize * ksize;
    int N = num_filters;

    int batches[2] = {batch_0, batch_1};
    int Ms[2] = {batch_0 * h_col * w_col, batch_1 * h_col * w_col};

    float *d_im[2], *d_col[2], *d_w[2], *d_out[2], *d_grad_w[2];

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        cudaMalloc(&d_im[g], batches[g] * img_h * img_w * sizeof(float));
        cudaMalloc(&d_col[g], Ms[g] * K * sizeof(float));
        cudaMalloc(&d_w[g], K * N * sizeof(float));
        cudaMalloc(&d_out[g], Ms[g] * N * sizeof(float));
        cudaMalloc(&d_grad_w[g], K * N * sizeof(float));
    }

    int threads = 256;
    dim3 block_gemm(TILE_SIZE, TILE_SIZE);

    // Warmup
    #pragma omp parallel for num_threads(2)
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        int blocks_im2col = (Ms[g] * K + threads - 1) / threads;
        dim3 grid_fwd((N + TILE_SIZE - 1) / TILE_SIZE, (Ms[g] + TILE_SIZE - 1) / TILE_SIZE);
        dim3 grid_bwd_w((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
        dim3 grid_bwd_x((K + TILE_SIZE - 1) / TILE_SIZE, (Ms[g] + TILE_SIZE - 1) / TILE_SIZE);

        im2col_batched_kernel<<<blocks_im2col, threads>>>(d_im[g], batches[g], 1, img_h, img_w, ksize, stride, h_col, w_col, d_col[g]);
        gemm_tiled_kernel<<<grid_fwd, block_gemm>>>(d_col[g], d_w[g], d_out[g], Ms[g], K, N);
        gemm_tiled_kernel<<<grid_bwd_w, block_gemm>>>(d_col[g], d_out[g], d_grad_w[g], K, Ms[g], N);
        gemm_tiled_kernel<<<grid_bwd_x, block_gemm>>>(d_out[g], d_w[g], d_col[g], Ms[g], N, K);
        cudaDeviceSynchronize();
    }

    double acc_gpu0_ms = 0.0;
    double acc_gpu1_ms = 0.0;
    double acc_total_ms = 0.0;

    for (int r = 0; r < num_runs; ++r) {
        double g_ms[2] = {0.0, 0.0};
        auto step_start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for num_threads(2)
        for (int g = 0; g < 2; ++g) {
            cudaSetDevice(g);
            int blocks_im2col = (Ms[g] * K + threads - 1) / threads;
            dim3 grid_fwd((N + TILE_SIZE - 1) / TILE_SIZE, (Ms[g] + TILE_SIZE - 1) / TILE_SIZE);
            dim3 grid_bwd_w((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
            dim3 grid_bwd_x((K + TILE_SIZE - 1) / TILE_SIZE, (Ms[g] + TILE_SIZE - 1) / TILE_SIZE);

            auto t_g0 = std::chrono::high_resolution_clock::now();
            im2col_batched_kernel<<<blocks_im2col, threads>>>(d_im[g], batches[g], 1, img_h, img_w, ksize, stride, h_col, w_col, d_col[g]);
            gemm_tiled_kernel<<<grid_fwd, block_gemm>>>(d_col[g], d_w[g], d_out[g], Ms[g], K, N);
            gemm_tiled_kernel<<<grid_bwd_w, block_gemm>>>(d_col[g], d_out[g], d_grad_w[g], K, Ms[g], N);
            gemm_tiled_kernel<<<grid_bwd_x, block_gemm>>>(d_out[g], d_w[g], d_col[g], Ms[g], N, K);
            cudaDeviceSynchronize();
            auto t_g1 = std::chrono::high_resolution_clock::now();
            g_ms[g] = std::chrono::duration<double, std::milli>(t_g1 - t_g0).count();
        }
        auto step_end = std::chrono::high_resolution_clock::now();
        double tot_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        acc_gpu0_ms += g_ms[0];
        acc_gpu1_ms += g_ms[1];
        acc_total_ms += tot_ms;
    }

    *out_gpu0_ms = acc_gpu0_ms / num_runs;
    *out_gpu1_ms = acc_gpu1_ms / num_runs;
    *out_total_ms = acc_total_ms / num_runs;
    *out_gpu0_idle_ms = std::max(0.0, *out_gpu1_ms - *out_gpu0_ms);

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        cudaFree(d_im[g]);
        cudaFree(d_col[g]);
        cudaFree(d_w[g]);
        cudaFree(d_out[g]);
        cudaFree(d_grad_w[g]);
    }
}

// Model Parallelism / Pipeline Parallelism across Dual GPUs (GPU 0: Conv1, GPU 1: Stage 2 + FC)
// Uses pinned host staging memory for PCIe transfers (robust across all consumer/datacenter GPUs)
double benchmark_model_parallelism_multigpu(int batch_size, int num_runs) {
    int img_h = 224, img_w = 224, ksize = 11, stride = 4, num_filters = 96;
    int h_col = (img_h - ksize) / stride + 1; // 54
    int w_col = (img_w - ksize) / stride + 1; // 54
    int M1 = batch_size * h_col * w_col;      // batch_size * 2916
    int K1 = ksize * ksize;                   // 121
    int N1 = num_filters;                     // 96

    int M2 = batch_size * 16;
    int K2 = num_filters;
    int N2 = 128;

    size_t transfer_bytes = static_cast<size_t>(M1) * N1 * sizeof(float);

    // Allocations on GPU 0 (Stage 1)
    cudaSetDevice(0);
    float *d_im0, *d_col0, *d_w0, *d_out0, *d_grad0;
    cudaMalloc(&d_im0, batch_size * img_h * img_w * sizeof(float));
    cudaMalloc(&d_col0, M1 * K1 * sizeof(float));
    cudaMalloc(&d_w0, K1 * N1 * sizeof(float));
    cudaMalloc(&d_out0, transfer_bytes);
    cudaMalloc(&d_grad0, transfer_bytes);

    // Allocations on GPU 1 (Stage 2)
    cudaSetDevice(1);
    float *d_in1, *d_w1, *d_out1, *d_grad1;
    cudaMalloc(&d_in1, transfer_bytes);
    cudaMalloc(&d_w1, K2 * N2 * sizeof(float));
    cudaMalloc(&d_out1, M2 * N2 * sizeof(float));
    cudaMalloc(&d_grad1, transfer_bytes);

    // Pinned host buffer for PCIe D2H -> H2D pipeline transfer
    float *h_stage_fwd, *h_stage_bwd;
    cudaHostAlloc(&h_stage_fwd, transfer_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_stage_bwd, transfer_bytes, cudaHostAllocDefault);

    int threads = 256;
    int blocks_im2col = (M1 * K1 + threads - 1) / threads;
    dim3 block_gemm(TILE_SIZE, TILE_SIZE);
    dim3 grid_fwd1((N1 + TILE_SIZE - 1) / TILE_SIZE, (M1 + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_fwd2((N2 + TILE_SIZE - 1) / TILE_SIZE, (M2 + TILE_SIZE - 1) / TILE_SIZE);

    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        cudaSetDevice(0);
        im2col_batched_kernel<<<blocks_im2col, threads, 0, stream0>>>(d_im0, batch_size, 1, img_h, img_w, ksize, stride, h_col, w_col, d_col0);
        gemm_tiled_kernel<<<grid_fwd1, block_gemm, 0, stream0>>>(d_col0, d_w0, d_out0, M1, K1, N1);
        cudaMemcpyAsync(h_stage_fwd, d_out0, transfer_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream0);

        cudaSetDevice(1);
        cudaMemcpyAsync(d_in1, h_stage_fwd, transfer_bytes, cudaMemcpyHostToDevice, stream1);
        gemm_tiled_kernel<<<grid_fwd2, block_gemm, 0, stream1>>>(d_in1, d_w1, d_out1, M2, K2, N2);
        cudaMemcpyAsync(h_stage_bwd, d_grad1, transfer_bytes, cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        cudaSetDevice(0);
        cudaMemcpyAsync(d_grad0, h_stage_bwd, transfer_bytes, cudaMemcpyHostToDevice, stream0);
        gemm_tiled_kernel<<<grid_fwd1, block_gemm, 0, stream0>>>(d_col0, d_out0, d_w0, K1, M1, N1);
        cudaStreamSynchronize(stream0);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < num_runs; ++r) {
        // 1. Stage 1 Forward on GPU 0
        cudaSetDevice(0);
        im2col_batched_kernel<<<blocks_im2col, threads, 0, stream0>>>(d_im0, batch_size, 1, img_h, img_w, ksize, stride, h_col, w_col, d_col0);
        gemm_tiled_kernel<<<grid_fwd1, block_gemm, 0, stream0>>>(d_col0, d_w0, d_out0, M1, K1, N1);
        cudaMemcpyAsync(h_stage_fwd, d_out0, transfer_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream0);

        // 2. Transfer to Stage 2 and Forward on GPU 1
        cudaSetDevice(1);
        cudaMemcpyAsync(d_in1, h_stage_fwd, transfer_bytes, cudaMemcpyHostToDevice, stream1);
        gemm_tiled_kernel<<<grid_fwd2, block_gemm, 0, stream1>>>(d_in1, d_w1, d_out1, M2, K2, N2);

        // 3. Backward on GPU 1 and Transfer Grad back to GPU 0
        cudaMemcpyAsync(h_stage_bwd, d_grad1, transfer_bytes, cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        // 4. Backward on GPU 0
        cudaSetDevice(0);
        cudaMemcpyAsync(d_grad0, h_stage_bwd, transfer_bytes, cudaMemcpyHostToDevice, stream0);
        gemm_tiled_kernel<<<grid_fwd1, block_gemm, 0, stream0>>>(d_col0, d_out0, d_w0, K1, M1, N1);
        cudaStreamSynchronize(stream0);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_runs;

    cudaSetDevice(0);
    cudaFree(d_im0); cudaFree(d_col0); cudaFree(d_w0); cudaFree(d_out0); cudaFree(d_grad0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_in1); cudaFree(d_w1); cudaFree(d_out1); cudaFree(d_grad1);
    cudaStreamDestroy(stream1);

    cudaFreeHost(h_stage_fwd);
    cudaFreeHost(h_stage_bwd);

    return avg_ms;
}

} // extern "C"
