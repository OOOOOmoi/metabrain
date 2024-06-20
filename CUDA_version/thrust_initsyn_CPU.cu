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

int main(){
    int scale=10;
    int numExc = scale*4096;
    float connect_prob = 0.02;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));

    ExponentSynapse E2E;
    auto start_init = high_resolution_clock::now();
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                E2E.pre.push_back(i);
                E2E.post.push_back(j);
                E2E.s.push_back(0.0);
                counter++;
            }
        }
    }
    auto end_init = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init - start_init);
    int numExc2Exc = counter;
    cout<<"Number of synapses: "<<counter<<endl;
    cout << "initTime: " << duration.count() << " ms" << endl;
}