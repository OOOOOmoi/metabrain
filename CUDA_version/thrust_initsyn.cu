#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <random>
#include <chrono>
#include <time.h>

using namespace std;
using namespace std::chrono;
struct ExponentSynapse {
    thrust::device_vector<int> pre;
    thrust::device_vector<int> post;
    thrust::device_vector<float> s;
};

__global__ void initializeSynapses(int numExc, float connect_prob, float* d_rand_values, int* d_pre, int* d_post, float* d_s, int* d_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numExc * numExc) {
        int i = idx / numExc;
        int j = idx % numExc;

        if (d_rand_values[idx] < connect_prob) {
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
    auto start_init = high_resolution_clock::now();

    // Allocate memory for the random values on host
    float* h_rand_values = new float[numExc * numExc];

    // Generate random values on host
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < numExc * numExc; i++) {
        h_rand_values[i] = dis(generator);
    }

    // Allocate memory on device
    float* d_rand_values;
    cudaMalloc((void**)&d_rand_values, numExc * numExc * sizeof(float));

    // Copy random values to device
    cudaMemcpy(d_rand_values, h_rand_values, numExc * numExc * sizeof(float), cudaMemcpyHostToDevice);

    // Cleanup host memory
    delete[] h_rand_values;

    // Define grid and block size
    int threadsPerBlock = 256;
    int blocksPerGrid = (numExc * numExc + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for the synapse data on device
    int maxSynapses = numExc * numExc;  // This is the maximum possible number of synapses
    int* d_pre, *d_post;
    float* d_s;
    int* d_counter;
    cudaMalloc((void**)&d_pre, maxSynapses * sizeof(int));
    cudaMalloc((void**)&d_post, maxSynapses * sizeof(int));
    cudaMalloc((void**)&d_s, maxSynapses * sizeof(float));
    cudaMalloc((void**)&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Launch kernel
    initializeSynapses<<<blocksPerGrid, threadsPerBlock>>>(numExc, connect_prob, d_rand_values, d_pre, d_post, d_s, d_counter);

    // Copy the counter value back to host to know the actual number of synapses
    int h_counter;
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy the synapse data back to host
    E2E.pre.resize(h_counter);
    E2E.post.resize(h_counter);
    E2E.s.resize(h_counter);

    thrust::copy(thrust::device_pointer_cast(d_pre), thrust::device_pointer_cast(d_pre) + h_counter, E2E.pre.begin());
    thrust::copy(thrust::device_pointer_cast(d_post), thrust::device_pointer_cast(d_post) + h_counter, E2E.post.begin());
    thrust::copy(thrust::device_pointer_cast(d_s), thrust::device_pointer_cast(d_s) + h_counter, E2E.s.begin());
    auto end_init = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init - start_init);
    // Cleanup device memory
    cudaFree(d_rand_values);
    cudaFree(d_pre);
    cudaFree(d_post);
    cudaFree(d_s);
    cudaFree(d_counter);

    cout << "Number of synapses: " << h_counter << endl;
    cout << "initTime: " << duration.count() << " ms" << endl;
    return 0;
}
