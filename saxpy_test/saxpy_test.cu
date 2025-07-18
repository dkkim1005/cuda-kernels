#include <vector>
#include "../include/common.hpp"


__global__ void warmup() {
}

__global__ void saxpy_with_all_threads(const float* x, float* y, const size_t N, const float alpha, const float beta) {
    register const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

__global__ void saxpy_with_small_threads(const float* x, float* y, const size_t N, const float alpha, const float beta) {
    register int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < N) {
        y[idx] = alpha * x[idx] + beta * y[idx];
        idx += gridDim.x * blockDim.x;
    }
}

__host__ void saxpy_host(const float* x, float* y, const size_t N, const float alpha, const float beta) {
    for (size_t i = 0u; i < N; ++i) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

int main() {
    constexpr int N = 16384 * 16384;
    constexpr float alpha = 1.0f, beta = 1.0f;
    std::vector<float> x(N), y1(N), y2(N), y3(N);
    for (int i = 0; i < N; ++i) {
        x[i] = 2 * (i + 1);
        y1[i] = 2 * i + 1;
        y2[i] = y1[i];
        y3[i] = y1[i];
    }

    CudaTimer<3> timer;
    float* dev_x = NULL;
    float* dev_y1 = NULL;
    float* dev_y2 = NULL;
    CUDA_CHECK(cudaMalloc(&dev_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&dev_y1, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&dev_y2, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(dev_x, &x[0], sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_y1, &y1[0], sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_y2, &y2[0], sizeof(float) * N, cudaMemcpyHostToDevice));

    // warm up before measurements
    warmup<<<1, 1>>>();

    timer.start(0);
    saxpy_with_all_threads<<<ceil((float)N / 1024), 1024>>>(dev_x, dev_y1, N, alpha, beta);
    // CUDA_CHECK(cudaMemcpy(&y1[0], dev_y1, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop(0);
    std::cout << std::setw(30) << "[saxpy_with_all_threads]: " << std::setw(10) << timer.meas(0) << " [ms] " << std::endl << std::flush;

    timer.start(1);
    saxpy_with_small_threads<<<1, 1024>>>(dev_x, dev_y2, N, alpha, beta);
    // CUDA_CHECK(cudaMemcpy(&y2[0], dev_y2, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop(1);
    std::cout << std::setw(30) << "[saxpy_with_small_threads]: " << std::setw(10) << timer.meas(1) << " [ms] " << std::endl << std::flush;

    timer.start(2);
    saxpy_host(&x[0], &y3[0], N, alpha, beta);
    timer.stop(2);
    std::cout << std::setw(30) << "[saxpy_host]: " << std::setw(10) << timer.meas(2) << " [ms] " << std::endl << std::flush;

    CUDA_CHECK(cudaFree(dev_x));
    CUDA_CHECK(cudaFree(dev_y1));
    CUDA_CHECK(cudaFree(dev_y2));

    return EXIT_SUCCESS;
}
