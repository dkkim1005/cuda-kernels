#include <vector>
#include <stdint.h>

#include "../include/common.hpp"


constexpr size_t N = 16384 * 16384;

__global__ void warmup() {
    printf("warming up for CUDA kernels\n");
}

#define WARM_UP_GPU() do {\
    warmup<<<1, 1>>>();\
    cudaDeviceSynchronize();\
} while (0)


__global__ void reduce_kernel1(const float* x, const size_t N, float* sum) {
    __shared__ float s_x[1024];

    size_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx < N) {
        s_x[threadIdx.x] = x[global_idx];
    } else {
        s_x[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (uint32_t offset = 512u; offset > 0u; offset /= 2u) {
        if (threadIdx.x < offset) {
            s_x[threadIdx.x] += s_x[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, s_x[0]);
    }
}


__global__ void reduce_kernel2(const float* x, const size_t N, float* sum) {
    __shared__ float s_x[32];  // Shared memory for 32 warp sums (1024 / 32)

    size_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = (global_idx < N) ? x[global_idx] : 0.0f;

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // First thread of each warp writes its partial sum to shared memory
    if (threadIdx.x % 32 == 0) {
        s_x[threadIdx.x / 32] = val;
    }

    __syncthreads();  // Synchronize to ensure all warp sums are written

    // First warp reduces the 32 partial sums
    if (threadIdx.x < 32) {
        val = s_x[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum, val);  // Thread 0 adds block sum to global sum
        }
    }
}


int main() {
    std::vector<float> x(N, 1.0f);
    float sum = 0.0f;
    float* x_dev, * sum_dev;
    CUDA_CHECK(cudaMalloc(&x_dev, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&sum_dev, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x_dev, &x[0], sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sum_dev, 0.0f, sizeof(float)));

    WARM_UP_GPU();
    const dim3 dimBlock(1024, 1, 1);
    const dim3 dimGrid((int)ceil((float)N / dimBlock.x), 1, 1);

    CudaTimer<2> timer;
    timer.start(0);
    reduce_kernel1<<<dimGrid, dimBlock>>>(x_dev, N, sum_dev);
    timer.stop(0);
    std::cout << "[" << std::setw(15) << "reduce_kernel1]: " << std::setw(10) << timer.meas(0) << " [ms] " << std::endl << std::flush;

    timer.start(1);
    reduce_kernel2<<<dimGrid, dimBlock>>>(x_dev, N, sum_dev);
    timer.stop(1);
    std::cout << "[" << std::setw(15) << "reduce_kernel2]: " << std::setw(10) << timer.meas(1) << " [ms] " << std::endl << std::flush;

    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(sum_dev));

    return EXIT_SUCCESS;
}
