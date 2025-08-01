#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Add elements of two arrays, result is stored in y.
__global__ void add(int numel, float *x, float *y) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = startIdx; i < numel; i += stride) {
        y[i] += x[i];
    }
}

int main() {
    int N = 1 << 20;
    float *x, *y;

    // Allocate inputs on unified-memory.
    cudaMallocManaged(&x, sizeof(float) * N);
    cudaMallocManaged(&y, sizeof(float) * N);

    // Initialize inputs.
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Prefetch the x and y arrays to the GPU
    cudaMemPrefetchAsync(x, N * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(y, N * sizeof(float), 0, 0);

    int blockSize = 256;
    int numBlocks = (N - 1) / blockSize + 1;

    // Run kernel on host.
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    // Assert result.
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i] - 3.0f));
    }
    printf("max error: %.4f\n", max_error);

    cudaFree(x);
    cudaFree(y);
}
