#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <random>
#include <vector>
#include <chrono>
#include <time.h>

using namespace std;
using namespace std::chrono;

struct ExponentSynapse {
    vector<int> pre;
    vector<int> post;
    vector<float> s;
};

__global__ void initializeSynapses(int numExc, int blockSize, float connect_prob, float* d_rand_values, int* d_pre, int* d_post, float* d_s, int* d_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < blockSize * blockSize) {
        int i = blockIdx.y * blockSize + idx / blockSize;
        int j = blockIdx.z * blockSize + idx % blockSize;

        if (i < numExc && j < numExc && d_rand_values[idx] < connect_prob) {
            int pos = atomicAdd(d_counter, 1);
            d_pre[pos] = i;
            d_post[pos] = j;
            d_s[pos] = 0.0;
        }
    }
}

int main() {
    int scale=10;
    int numExc = scale*4096;
    float connect_prob = 0.02;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    ExponentSynapse E2E;

    // Define block size
    int blockSize = 512;

    // Allocate memory for the counter on device
    int* d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));

    // Allocate memory for the random values on device
    float* d_rand_values;
    cudaMalloc((void**)&d_rand_values, blockSize * blockSize * sizeof(float));
    auto start_init = high_resolution_clock::now();
    for (int block_i = 0; block_i < (numExc + blockSize - 1) / blockSize; ++block_i) {
        for (int block_j = 0; block_j < (numExc + blockSize - 1) / blockSize; ++block_j) {
            // Allocate memory for the random values on host
            float* h_rand_values = new float[blockSize * blockSize];

            // Generate random values on host
            std::uniform_real_distribution<float> dis(0.0, 1.0);
            for (int i = 0; i < blockSize * blockSize; i++) {
                h_rand_values[i] = dis(generator);
            }

            // Copy random values to device
            cudaMemcpy(d_rand_values, h_rand_values, blockSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);

            // Cleanup host memory
            delete[] h_rand_values;

            // Allocate memory for the synapse data on device
            int batch_size = blockSize * blockSize;
            int* d_pre, *d_post;
            float* d_s;
            cudaMalloc((void**)&d_pre, batch_size * sizeof(int));
            cudaMalloc((void**)&d_post, batch_size * sizeof(int));
            cudaMalloc((void**)&d_s, batch_size * sizeof(float));

            // Reset counter on device
            cudaMemset(d_counter, 0, sizeof(int));

            // Define grid and block size
            dim3 threadsPerBlock(256);
            dim3 blocksPerGrid((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, block_i, block_j);
            
            // Launch kernel
            initializeSynapses<<<blocksPerGrid, threadsPerBlock>>>(numExc, blockSize, connect_prob, d_rand_values, d_pre, d_post, d_s, d_counter);

            // Copy the counter value back to host to know the actual number of synapses in this batch
            int h_counter;
            cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

            // Copy the synapse data back to host
            int* h_pre = new int[h_counter];
            int* h_post = new int[h_counter];
            float* h_s = new float[h_counter];

            cudaMemcpy(h_pre, d_pre, h_counter * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_post, d_post, h_counter * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_s, d_s, h_counter * sizeof(float), cudaMemcpyDeviceToHost);

            // Append to the ExponentSynapse structure
            E2E.pre.insert(E2E.pre.end(), h_pre, h_pre + h_counter);
            E2E.post.insert(E2E.post.end(), h_post, h_post + h_counter);
            E2E.s.insert(E2E.s.end(), h_s, h_s + h_counter);

            // Cleanup host memory
            delete[] h_pre;
            delete[] h_post;
            delete[] h_s;

            // Cleanup device memory
            cudaFree(d_pre);
            cudaFree(d_post);
            cudaFree(d_s);
        }
    }
    auto end_init = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init - start_init);
    // Cleanup device memory
    cudaFree(d_rand_values);
    cudaFree(d_counter);

    cout << "Number of synapses: " << E2E.pre.size() << endl;
    cout << "initTime: " << duration.count() << " ms" << endl;
    return 0;
}
