#include <vector>
#include "../include/common.hpp"


constexpr int STREAM_SIZE = 8;
constexpr int N = 16384 * 16384;

__global__ void warmup() {
    printf("warming up for CUDA kernels\n");
}

__global__ void saxpy_kernel(const float* x, float* y, const size_t N, const float alpha) {
    register const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

__host__ void saxpy_naive(const float* x, float* y, const size_t N, const float alpha) {
    float* dev_x = NULL;
    float* dev_y = NULL;

    CUDA_CHECK(cudaMalloc(&dev_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&dev_y, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(dev_x, x, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_y, y, sizeof(float) * N, cudaMemcpyHostToDevice));

    saxpy_kernel<<<ceil((float)N / 1024), 1024>>>(dev_x, dev_y, N, alpha);
    CUDA_CHECK(cudaMemcpy(&y[0], dev_y, sizeof(float) * N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev_x));
    CUDA_CHECK(cudaFree(dev_y));
}

template <int NUM_STREAMS>
__host__ void saxpy_stream(const float* x, float* y, const size_t N, const float alpha) {
    float* dev_x = NULL;
    float* dev_y = NULL;
    cudaStream_t stream[NUM_STREAMS];

    CUDA_CHECK(cudaMalloc(&dev_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&dev_y, sizeof(float) * N));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    const int chunkSize = N / NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int offset = chunkSize * i;
        CUDA_CHECK(cudaMemcpyAsync(dev_x + offset, x + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, stream[i]));
        CUDA_CHECK(cudaMemcpyAsync(dev_y + offset, y + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, stream[i]));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int offset = chunkSize * i;
        saxpy_kernel<<<(int)ceil((float)chunkSize / 1024), 1024, 0, stream[i]>>>(dev_x + offset, dev_y + offset, chunkSize, alpha);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int offset = chunkSize * i;
        CUDA_CHECK(cudaMemcpyAsync(y + offset, dev_y + offset, sizeof(float) * chunkSize, cudaMemcpyDeviceToHost, stream[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    cudaFree(dev_x);
    cudaFree(dev_y);
}

__host__ void saxpy_host(const float* x, float* y, const size_t N, const float alpha) {
    for (size_t i = 0u; i < N; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

__host__ bool check_distance(const float* y1, const float* y2, const size_t N) {
    double accum = 0.0;
    for (int i = 0; i < N; ++i) {
        accum += std::abs(y1[i] - y2[i]);
    }
    accum /= N;
    return accum < 1e-7;
}


int main() {
    constexpr float alpha = 1.0f;
    std::vector<float> x_cpu(N), y_cpu(N);
    float *x_host, *y1_host, *y2_host, *y3_host;
    cudaMallocHost(&x_host, sizeof(float) * N);
    cudaMallocHost(&y1_host, sizeof(float) * N);
    cudaMallocHost(&y2_host, sizeof(float) * N);
    cudaMallocHost(&y3_host, sizeof(float) * N);
    for (int i = 0; i < N; ++i) {
        x_cpu[i] = 2 * (i + 1);
        y_cpu[i] = 2 * i + 1;
        x_host[i] = x_cpu[i];
        y1_host[i] = y_cpu[i];
        y2_host[i] = y_cpu[i];
        y3_host[i] = y_cpu[i];
    }

    CudaTimer<4> timer;

    // warm up before measurements
    warmup<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.start(0);
    saxpy_host(&x_cpu[0], &y_cpu[0], N, alpha);
    timer.stop(0);
    std::cout << "[" << std::setw(15) << "saxpy_cpu]: " << std::setw(10) << timer.meas(0) << " [ms] " << std::endl << std::flush;

    timer.start(1);
    saxpy_host(&x_host[0], &y1_host[0], N, alpha);
    timer.stop(1);
    std::cout << "[" << std::setw(15) << "saxpy_host]: " << std::setw(10) << timer.meas(1) << " [ms] " << std::endl << std::flush;

    timer.start(2);
    saxpy_naive(&x_host[0], &y2_host[0], N, alpha);
    timer.stop(2);
    if (check_distance(y1_host, y2_host, N)) {
        std::cout << "[" << std::setw(15) << "saxpy_naive]: " << std::setw(10) << timer.meas(2) << " [ms] " << std::endl << std::flush;
    } else {
        std::cout << "[" << std::setw(15) << "saxpy_naive]: " << "wrong results...!" << std::endl;
    }

    timer.start(3);
    saxpy_stream<STREAM_SIZE>(&x_host[0], &y3_host[0], N, alpha);
    timer.stop(3);
    if (check_distance(y1_host, y3_host, N)) {
        std::cout << "[" << std::setw(15) << "saxpy_stream]: " << std::setw(10) << timer.meas(3) << " [ms] " << std::endl << std::flush;
    } else {
        std::cout << "[" << std::setw(15) << "saxpy_stream]: " << "wrong results...!" << std::endl;
    }

    cudaFreeHost(x_host);
    cudaFreeHost(y1_host);
    cudaFreeHost(y2_host);
    cudaFreeHost(y3_host);

    return EXIT_SUCCESS;
}
