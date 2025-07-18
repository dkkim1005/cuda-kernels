#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>

#define CUDA_CHECK(errMsg) do {\
    if (errMsg != cudaSuccess) {\
        std::cerr << "CUDA ERROR MSG: " << cudaGetErrorString(errMsg) << " (LINE: " << __LINE__ << ")" << std::endl;\
        exit(1);\
    }\
} while (0)


template <int N = 1>
class CudaTimer {
  public:
    CudaTimer() {
        for (int i = 0; i < N; ++i) {
            cudaEventCreate(&start_[i]);
            cudaEventCreate(&stop_[i]);
        }
    }

    ~CudaTimer() {
        for (int i = 0; i < N; ++i) {
            cudaEventDestroy(start_[i]);
            cudaEventDestroy(stop_[i]);
        }
    }

    void start(const int i = 0) {
        assert(i < N);
        cudaEventRecord(start_[i]);
    };

    void stop(const int i = 0) {
        assert(i < N);
        cudaEventRecord(stop_[i]);
    };

    float meas(const int i = 0) {
        assert(i < N);
        cudaEventSynchronize(stop_[i]);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_[i], stop_[i]);
        return milliseconds;
    }

  private:
    cudaEvent_t start_[N];
    cudaEvent_t stop_[N];
};
