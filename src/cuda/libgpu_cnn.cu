/**
 * Native CUDA C++ Bridge for PyTorch on sm_61 (GTX 1080 Ti & GTX 1080)
 * Provides C-linkage GPU functions for:
 * - Multi-GPU (1x GPU and 2x GPU DataParallel via OpenMP threads across cudaSetDevice(0/1))
 *   forward + backward CNN step timing (`im2colgpu` + GEMM).
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
        int row_idx = index / patch_len; // [0 .. batch_size * patches_per_img - 1]
        int col_idx = index % patch_len; // [0 .. channels * ksize * ksize - 1]

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

// Benchmarks a full CNN Forward + Backward minibatch step (batch_size=64) across num_gpus (1 or 2)
double benchmark_cnn_step_multigpu(int batch_size, int num_gpus, int num_runs) {
    int local_bs = batch_size / num_gpus;
    int img_h = 224, img_w = 224, ksize = 11, stride = 4, num_filters = 96;
    int h_col = (img_h - ksize) / stride + 1; // 54
    int w_col = (img_w - ksize) / stride + 1; // 54
    int M = local_bs * h_col * w_col;         // local_bs * 2916
    int K = ksize * ksize;                    // 121
    int N = num_filters;                      // 96

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
            // Forward pass (im2colgpu + GEMM)
            im2col_batched_kernel<<<blocks_im2col, threads>>>(d_im[g], local_bs, 1, img_h, img_w, ksize, stride, h_col, w_col, d_col[g]);
            gemm_tiled_kernel<<<grid_fwd, block_gemm>>>(d_col[g], d_w[g], d_out[g], M, K, N);
            // Backward pass (dW and dX GEMMs)
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

} // extern "C"
