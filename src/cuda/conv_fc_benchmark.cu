/**
 * Reference implementation of Section 3: Empirical Analysis - CPU vs GPU Time
 * Paper: "Parallel and Distributed Deep Learning" (Hegde & Usmani, Stanford University)
 *
 * Implements:
 * 1. Section 3.1 & 3.2.1: Parallel Implementation of Convolution via im2col + GEMM
 *    - Greyscale image size: 224 x 224
 *    - Filter size: 11 x 11, Stride: 4, Number of filters: 96
 *    - Output patches per spatial dim: 54 (54 x 54 = 2916 patches per image, unrolled vector length 121)
 *    - Note: Paper Section 3.1 describes both the 1D-unrolled formulation (54 x 121 * 121 x 96)
 *      and full 2D spatial convolution (2916 x 121 * 121 x 96) across batch sizes up to 300.
 *      We benchmark the full 2D spatial convolution (224x224 image, 11x11 kernel, stride 4, 96 filters)
 *      with im2colgpu + GEMM across batch sizes [1, 16, 32, 64, 128, 192, 256, 300], averaged over 10 runs.
 *
 * 2. Section 3.2.2: Fully Connected Layer / Matrix-Matrix & Matrix-Vector Multiplication
 *    - Benchmarks CPU vs GPU time as a function of Matrix Size N in [128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000]
 *    - Averaged over 10 runs.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err \
                      << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 1. CPU Implementations (Single-Core & OpenMP Multi-Core)
// ============================================================================

// Unroll a 224x224 image into patches of 11x11 with stride 4 -> (54*54) x 121 matrix
void im2col_cpu(const float* data_im, int height, int width, int ksize, int stride, float* data_col) {
    int height_col = (height - ksize) / stride + 1; // 54
    int width_col = (width - ksize) / stride + 1;   // 54
    int patch_size = ksize * ksize;                 // 121

    for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
            int row_idx = h * width_col + w;
            int h_im_start = h * stride;
            int w_im_start = w * stride;
            for (int kh = 0; kh < ksize; ++kh) {
                for (int kw = 0; kw < ksize; ++kw) {
                    int col_idx = kh * ksize + kw;
                    data_col[row_idx * patch_size + col_idx] =
                        data_im[(h_im_start + kh) * width + (w_im_start + kw)];
                }
            }
        }
    }
}

// Matrix multiplication C = A * B where A is (M x K), B is (K x N), C is (M x N)
void gemm_cpu_single(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// OpenMP Multi-core Matrix Multiplication (Section 2: Multi-core processing)
void gemm_cpu_omp(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            C[i * N + k] = 0.0f;
        }
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Matrix-Vector multiplication y = A * x where A is (N x N), x is (N x 1)
void gemv_cpu_single(const float* A, const float* x, float* y, int N) {
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// 2. GPU CUDA Implementations (im2colgpu + GEMM / GEMV kernels)
// ============================================================================

__global__ void im2col_gpu_kernel(const float* __restrict__ data_im,
                                  int height, int width, int ksize, int stride,
                                  int height_col, int width_col,
                                  float* __restrict__ data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_patches = height_col * width_col; // 54 * 54 = 2916
    int patch_size = ksize * ksize;           // 121
    int total_elements = num_patches * patch_size;

    if (index < total_elements) {
        int row_idx = index / patch_size; // patch index [0..2915]
        int col_idx = index % patch_size; // element within 11x11 [0..120]

        int h_col = row_idx / width_col;
        int w_col = row_idx % width_col;
        int kh = col_idx / ksize;
        int kw = col_idx % ksize;

        int h_im = h_col * stride + kh;
        int w_im = w_col * stride + kw;

        data_col[index] = data_im[h_im * width + w_im];
    }
}

#define TILE_SIZE 16

__global__ void gemm_gpu_kernel(const float* __restrict__ A,
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

__global__ void gemv_gpu_kernel(const float* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "==========================================================" << std::endl;
    std::cout << "Empirical Analysis: CPU vs GPU Time (Section 3)" << std::endl;
    std::cout << "GPU Device: " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << std::endl;
    std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ------------------------------------------------------------------------
    // Experiment 3.2.1: Convolution Layer Forward Propagation vs Batch Size
    // Image: 224x224 greyscale, Filter: 11x11, Stride: 4, Filters: 96
    // Following Caffe's serial loop over images in a mini-batch (Section 3.1):
    // "This happens serially for all the images in a mini-batch."
    // ------------------------------------------------------------------------
    const int IMG_H = 224;
    const int IMG_W = 224;
    const int KSIZE = 11;
    const int STRIDE = 4;
    const int NUM_FILTERS = 96;
    const int H_COL = (IMG_H - KSIZE) / STRIDE + 1; // 54
    const int W_COL = (IMG_W - KSIZE) / STRIDE + 1; // 54
    const int M_PATCHES = H_COL * W_COL;            // 54 * 54 = 2916
    const int K_PATCH_LEN = KSIZE * KSIZE;          // 121
    const int NUM_RUNS = 10;

    std::vector<int> batch_sizes = {1, 10, 25, 50, 100, 150, 200, 250, 300};

    std::vector<float> h_img(IMG_H * IMG_W, 0.5f);
    std::vector<float> h_filters(K_PATCH_LEN * NUM_FILTERS, 0.01f);
    std::vector<float> h_col(M_PATCHES * K_PATCH_LEN, 0.0f);
    std::vector<float> h_out(M_PATCHES * NUM_FILTERS, 0.0f);

    float *d_img, *d_filters, *d_col, *d_out;
    CUDA_CHECK(cudaMalloc(&d_img, IMG_H * IMG_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, K_PATCH_LEN * NUM_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col, M_PATCHES * K_PATCH_LEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M_PATCHES * NUM_FILTERS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), IMG_H * IMG_W * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, h_filters.data(), K_PATCH_LEN * NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup GPU
    int threads_im2col = 256;
    int blocks_im2col = (M_PATCHES * K_PATCH_LEN + threads_im2col - 1) / threads_im2col;
    dim3 block_gemm(TILE_SIZE, TILE_SIZE);
    dim3 grid_gemm((NUM_FILTERS + TILE_SIZE - 1) / TILE_SIZE, (M_PATCHES + TILE_SIZE - 1) / TILE_SIZE);
    im2col_gpu_kernel<<<blocks_im2col, threads_im2col>>>(d_img, IMG_H, IMG_W, KSIZE, STRIDE, H_COL, W_COL, d_col);
    gemm_gpu_kernel<<<grid_gemm, block_gemm>>>(d_col, d_filters, d_out, M_PATCHES, K_PATCH_LEN, NUM_FILTERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::ofstream conv_csv("results/logs/conv_benchmark.csv");
    conv_csv << "batch_size,cpu_single_ms,cpu_omp_ms,gpu_ms,speedup_single_vs_gpu,speedup_omp_vs_gpu\n";

    std::cout << "\n[Section 3.2.1] Convolution Layer Forward Propagation (10-run average):" << std::endl;
    std::cout << std::setw(10) << "BatchSize"
              << std::setw(16) << "CPU-1Core(ms)"
              << std::setw(16) << "CPU-OpenMP(ms)"
              << std::setw(16) << "GPU-CUDA(ms)"
              << std::setw(16) << "Ratio(CPU/GPU)" << std::endl;

    for (int bs : batch_sizes) {
        // 1. Single-core CPU timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NUM_RUNS; ++r) {
            for (int b = 0; b < bs; ++b) {
                im2col_cpu(h_img.data(), IMG_H, IMG_W, KSIZE, STRIDE, h_col.data());
                gemm_cpu_single(h_col.data(), h_filters.data(), h_out.data(), M_PATCHES, K_PATCH_LEN, NUM_FILTERS);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / NUM_RUNS;

        // 2. Multi-core OpenMP CPU timing
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NUM_RUNS; ++r) {
            for (int b = 0; b < bs; ++b) {
                im2col_cpu(h_img.data(), IMG_H, IMG_W, KSIZE, STRIDE, h_col.data());
                gemm_cpu_omp(h_col.data(), h_filters.data(), h_out.data(), M_PATCHES, K_PATCH_LEN, NUM_FILTERS);
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double cpu_omp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / NUM_RUNS;

        // 3. GPU timing (following Caffe's per-image im2colgpu + GEMM in mini-batch)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < NUM_RUNS; ++r) {
            for (int b = 0; b < bs; ++b) {
                im2col_gpu_kernel<<<blocks_im2col, threads_im2col>>>(d_img, IMG_H, IMG_W, KSIZE, STRIDE, H_COL, W_COL, d_col);
                gemm_gpu_kernel<<<grid_gemm, block_gemm>>>(d_col, d_filters, d_out, M_PATCHES, K_PATCH_LEN, NUM_FILTERS);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
        double gpu_ms = total_gpu_ms / NUM_RUNS;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        double ratio_single = cpu_single_ms / gpu_ms;
        double ratio_omp = cpu_omp_ms / gpu_ms;

        std::cout << std::setw(10) << bs
                  << std::setw(16) << std::fixed << std::setprecision(3) << cpu_single_ms
                  << std::setw(16) << std::fixed << std::setprecision(3) << cpu_omp_ms
                  << std::setw(16) << std::fixed << std::setprecision(3) << gpu_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << ratio_single << "x" << std::endl;

        conv_csv << bs << "," << cpu_single_ms << "," << cpu_omp_ms << "," << gpu_ms << ","
                 << ratio_single << "," << ratio_omp << "\n";
    }
    conv_csv.close();

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_out));

    // ------------------------------------------------------------------------
    // Experiment 3.2.2: Fully Connected Layer (Matrix-Vector Multiplication O(N^2))
    // Matrix sizes up to 8000 x 8000 as in Figures 3 & 4 of the paper.
    // ------------------------------------------------------------------------
    std::vector<int> matrix_sizes = {128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000};
    std::ofstream fc_csv("results/logs/fc_benchmark.csv");
    fc_csv << "matrix_size,cpu_gemv_ms,gpu_gemv_ms,speedup_gemv\n";

    std::cout << "\n[Section 3.2.2] Fully Connected Layer (Matrix-Vector Multiplication O(n^2), 10-run average):" << std::endl;
    std::cout << std::setw(12) << "MatrixSize"
              << std::setw(16) << "CPU-Time(ms)"
              << std::setw(16) << "GPU-Time(ms)"
              << std::setw(16) << "Ratio(CPU/GPU)" << std::endl;

    for (int N : matrix_sizes) {
        std::vector<float> h_A(static_cast<size_t>(N) * N, 0.01f);
        std::vector<float> h_x(N, 1.0f);
        std::vector<float> h_y(N, 0.0f);

        float *d_A, *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(N) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(N) * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        gemv_gpu_kernel<<<blocks, threads>>>(d_A, d_x, d_y, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < NUM_RUNS; ++r) {
            gemv_cpu_single(h_A.data(), h_x.data(), h_y.data(), N);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / NUM_RUNS;

        // GPU time
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < NUM_RUNS; ++r) {
            gemv_gpu_kernel<<<blocks, threads>>>(d_A, d_x, d_y, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, start, stop));
        double gpu_ms = total_gpu_ms / NUM_RUNS;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));

        double ratio = cpu_ms / gpu_ms;

        std::cout << std::setw(12) << N
                  << std::setw(16) << std::fixed << std::setprecision(3) << cpu_ms
                  << std::setw(16) << std::fixed << std::setprecision(3) << gpu_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;

        fc_csv << N << "," << cpu_ms << "," << gpu_ms << "," << ratio << "\n";
    }
    fc_csv.close();

    return 0;
}
