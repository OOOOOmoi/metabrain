#include <iostream>
#include <cuda_runtime.h>
// #include "/home/yangjinhao/enlarge-backup/enlarge-input-myself/src/third_party/cuda/helper_cuda.h"
#include <random>
#include <chrono>
#include <vector>
#include <time.h>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
using namespace std;
using namespace std::chrono;

const int MAX_SPIKES = 10000; // 假设每个神经元最多记录1000次脉冲

struct LIFNeuron {
    thrust::device_vector<float> V;
    thrust::device_vector<float> refractory_time;
    thrust::device_vector<float> input_current;
    thrust::device_vector<bool> spiked;
};

struct ExponentialSynapse {
    thrust::device_vector<int> pre;  // 突触前神经元索引
    thrust::device_vector<int> post; // 突触后神经元索引
    thrust::device_vector<float> s;  // 突触的状态
};

__global__ void simulateNeuronsFixpara(float* iSyn, bool* Spike, float* Ref, float* Potential, \
                                        int num_neurons, int input, float dt, float* spike_times, \
                                        int* spike_counts, int time_step){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<num_neurons){
        iSyn[tid]+=input;
        Spike[tid] = false;
        if (Ref[tid] > 0) {
            Ref[tid] -= dt;
            Potential[tid] = -60.0;//reset voltage
        } else {
            //V_inf=E_L+RI;
            //V=V+dt*(V_inf-V)/tau_m
            //ouler
            float V_inf = -60.0 + 1.0 * iSyn[tid];//EL
            Potential[tid] += dt * (V_inf - Potential[tid]) / 20.0;//tau_m
            if (Potential[tid] >= -50.0) {//V_th
                Spike[tid] = true;
                Potential[tid] = -60.0;//V_reset
                Ref[tid] = 5.0;//tau_ref
            }
        }
        iSyn[tid] = 0;  // 重置输入电流为下一时间步准备
        if (Spike[tid]) {
            int temp = atomicAdd(&spike_counts[tid], 1);
            spike_times[tid * MAX_SPIKES + temp] = time_step * dt;
        }
    }
}

__global__ void simulateSynapsesFixparaAmpa(float* state, int* Pre, int* Post, bool* pre_spike, \
                                            float* post_input, float* post_V, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        if (pre_spike[Pre[tid]]) {
            state[tid] += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        state[tid] -= state[tid] / 5.0 * dt;  // s的指数衰减,tau
        float g_exp = 0.3 * state[tid];//g_max
        float I_syn = g_exp * (0.0 - post_V[Post[tid]]);  // 计算突触电流,EL
        atomicAdd(&post_input[Post[tid]], I_syn);  // 原子加，以避免并发写入问题
        // postneurons->input_current[synapses->post[tid]]+=I_syn;
    }
}

__global__ void simulateSynapsesFixparaGaba(float* state, int* Pre, int* Post, bool* pre_spike, \
                                            float* post_input, float* post_V, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        if (pre_spike[Pre[tid]]) {
            state[tid] += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        state[tid] -= state[tid] / 10.0 * dt;  // s的指数衰减,tau
        float g_exp = 3.2 * state[tid];//g_max
        float I_syn = g_exp * (-80.0 - post_V[Post[tid]]);  // 计算突触电流,EL
        atomicAdd(&post_input[Post[tid]], I_syn);  // 原子加，以避免并发写入问题
        // postneurons->input_current[synapses->post[tid]]+=I_syn;
    }
}


int save_spike(int* h_spike_counts_exc, float* h_spike_times_exc, int* h_spike_counts_inh, float* h_spike_times_inh, int numExc, int numInh) {
    FILE* exc_spike_file = fopen("exc_spike_times.txt", "w");
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < h_spike_counts_exc[i]; j++) {
            fprintf(exc_spike_file, "%f ", h_spike_times_exc[i * MAX_SPIKES + j]);
        }
        fprintf(exc_spike_file, "\n");
    }
    fclose(exc_spike_file);
    FILE* inh_spike_file = fopen("inh_spike_times.txt", "w");
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < h_spike_counts_inh[i]; j++) {
            fprintf(inh_spike_file, "%f ", h_spike_times_inh[i * MAX_SPIKES + j]);
        }
        fprintf(inh_spike_file, "\n");
    }
    fclose(inh_spike_file);
    return 0;
}

// 主函数，设置和运行模拟
int main() {
    int scale = 1;
    int numExc = 4096 * scale;
    int numInh = 1024 * scale;
    float connect_prob = 0.02;
    float dt = 0.1;
    int steps = 10000;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    //定义神经元群和突触连接
    // LIFNeuron *PopExc;
    LIFNeuron *d_PopExc;
    // LIFNeuron *PopInh;
    LIFNeuron *d_PopInh;

    // 初始化神经元参数
    auto start_init_neuron = high_resolution_clock::now();
    d_PopExc->V.resize(numExc,-60.0);
    d_PopExc->refractory_time.resize(numExc,0.0);
    d_PopExc->input_current.resize(numExc,0.0);
    d_PopExc->spiked.resize(numExc,false);
    float* PopExc_V_ptr=d_PopExc->V.data().get();
    float* PopExc_Ref_ptr=d_PopExc->refractory_time.data().get();
    float* PopExc_iSyn_ptr=d_PopExc->input_current.data().get();
    bool* PopExc_Spike_ptr=d_PopExc->spiked.data().get();

    d_PopInh->V.resize(numInh,-60.0);
    d_PopInh->refractory_time.resize(numInh,0.0);
    d_PopInh->input_current.resize(numInh,0.0);
    d_PopInh->spiked.resize(numInh,false);
    float* PopInh_V_ptr=d_PopInh->V.data().get();
    float* PopInh_Ref_ptr=d_PopInh->refractory_time.data().get();
    float* PopInh_iSyn_ptr=d_PopInh->input_current.data().get();
    bool* PopInh_Spike_ptr=d_PopInh->spiked.data().get();
    
    //

    auto end_init_neuron = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init_neuron - start_init_neuron);
    cout << "neurons initTime: " << duration.count() << " ms" << endl;

    // ExponentialSynapse* Exc2ExcSyn_AMPA;
    ExponentialSynapse *d_Exc2ExcSyn_AMPA;
    // ExponentialSynapse* Exc2InhSyn_AMPA;
    ExponentialSynapse *d_Exc2InhSyn_AMPA;
    // ExponentialSynapse* Inh2ExcSyn_GABA;
    ExponentialSynapse *d_Inh2ExcSyn_GABA;
    // ExponentialSynapse* Inh2InhSyn_GABA;
    ExponentialSynapse *d_Inh2InhSyn_GABA;

    auto start_init = high_resolution_clock::now();

    // 分配和初始化突触连接参数
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                d_Exc2ExcSyn_AMPA->pre.push_back(i);
                d_Exc2ExcSyn_AMPA->post.push_back(j);
                d_Exc2ExcSyn_AMPA->s.push_back(0.0);
                counter++;
            }
        }
    }
    int numExc2Exc = counter;
    float* E2E_state_ptr=d_Exc2ExcSyn_AMPA->s.data().get();
    int* E2E_PreIdx_ptr=d_Exc2ExcSyn_AMPA->pre.data().get();
    int* E2E_PostIdx_ptr=d_Exc2ExcSyn_AMPA->post.data().get();
    bool* E2E_pre_spike_ptr=d_PopExc->spiked.data().get();
    float* E2E_post_input_ptr=d_PopExc->input_current.data().get();
    float* E2E_post_V_ptr=d_PopExc->V.data().get();


    counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                d_Exc2InhSyn_AMPA->pre.push_back(i);
                d_Exc2InhSyn_AMPA->post.push_back(j);
                d_Exc2InhSyn_AMPA->s.push_back(0.0);
                counter++;
            }
        }
    }
    int numExc2Inh = counter;
    float* E2I_state_ptr=d_Exc2InhSyn_AMPA->s.data().get();
    int* E2I_PreIdx_ptr=d_Exc2InhSyn_AMPA->pre.data().get();
    int* E2I_PostIdx_ptr=d_Exc2InhSyn_AMPA->post.data().get();
    bool* E2I_pre_spike_ptr=d_PopExc->spiked.data().get();
    float* E2I_post_input_ptr=d_PopInh->input_current.data().get();
    float* E2I_post_V_ptr=d_PopInh->V.data().get();

    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                d_Inh2ExcSyn_GABA->pre.push_back(i);
                d_Inh2ExcSyn_GABA->post.push_back(j);
                d_Inh2ExcSyn_GABA->s.push_back(0.0);
                counter++;
            }
        }
    }
    int numInh2Exc = counter;
    float* I2E_state_ptr=d_Inh2ExcSyn_GABA->s.data().get();
    int* I2E_PreIdx_ptr=d_Inh2ExcSyn_GABA->pre.data().get();
    int* I2E_PostIdx_ptr=d_Inh2ExcSyn_GABA->post.data().get();
    bool* I2E_pre_spike_ptr=d_PopInh->spiked.data().get();
    float* I2E_post_input_ptr=d_PopExc->input_current.data().get();
    float* I2E_post_V_ptr=d_PopExc->V.data().get();


    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                d_Inh2InhSyn_GABA->pre.push_back(i);
                d_Inh2InhSyn_GABA->post.push_back(j);
                d_Inh2InhSyn_GABA->s.push_back(0.0);
                counter++;
            }
        }
    }
    int numInh2Inh = counter;
    float* I2I_state_ptr=d_Inh2InhSyn_GABA->s.data().get();
    int* I2I_PreIdx_ptr=d_Inh2InhSyn_GABA->pre.data().get();
    int* I2I_PostIdx_ptr=d_Inh2InhSyn_GABA->post.data().get();
    bool* I2I_pre_spike_ptr=d_PopInh->spiked.data().get();
    float* I2I_post_input_ptr=d_PopInh->input_current.data().get();
    float* I2I_post_V_ptr=d_PopInh->V.data().get();

    auto end_init = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_init - start_init);
    cout << "initTime: " << duration.count() << " ms" << endl;

    // 初始化GPU变量
    float *d_spike_times_exc, *d_spike_times_inh;
    int *d_spike_counts_exc, *d_spike_counts_inh;
    cudaMalloc(&d_spike_times_exc, numExc * MAX_SPIKES * sizeof(float));
    cudaMalloc(&d_spike_times_inh, numInh * MAX_SPIKES * sizeof(float));
    cudaMalloc(&d_spike_counts_exc, numExc * sizeof(int));
    cudaMalloc(&d_spike_counts_inh, numInh * sizeof(int));

    cudaMemset(d_spike_counts_exc, 0, numExc * sizeof(int));
    cudaMemset(d_spike_counts_inh, 0, numInh * sizeof(int));

    float *h_spike_times_exc = new float[numExc * MAX_SPIKES];
    float *h_spike_times_inh = new float[numInh * MAX_SPIKES];
    int *h_spike_counts_exc = new int[numExc];
    int *h_spike_counts_inh = new int[numInh];

    // 设置CUDA kernel的执行配置
    int threadsPerBlock = 1024;
    int blocksPerGridExc = (numExc + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridInh = (numInh + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridExc2Exc = (numExc2Exc + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridExc2Inh = (numExc2Inh + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridInh2Exc = (numInh2Exc + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridInh2Inh = (numInh2Inh + threadsPerBlock - 1) / threadsPerBlock;

    // 运行模拟
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int freq=20;
    float input = 12.0;
    for (int t = 0; t < steps; t++) {
        // int input = (sin(2 * 3.1415 * t / freq) + 1) * 10;
        simulateNeuronsFixpara<<<blocksPerGridExc, threadsPerBlock>>>(PopExc_iSyn_ptr, PopExc_Spike_ptr, PopExc_Ref_ptr, PopExc_V_ptr, numExc, input, dt, d_spike_times_exc, d_spike_counts_exc, t);
        simulateNeuronsFixpara<<<blocksPerGridInh, threadsPerBlock>>>(PopInh_iSyn_ptr, PopInh_Spike_ptr, PopInh_Ref_ptr, PopInh_V_ptr, numInh, input, dt, d_spike_times_inh, d_spike_counts_inh, t);
        cudaDeviceSynchronize();

        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Exc, threadsPerBlock>>>(E2E_state_ptr, E2E_PreIdx_ptr, E2E_PostIdx_ptr,E2E_pre_spike_ptr,E2E_post_input_ptr, E2E_post_V_ptr, numExc2Exc, dt);
        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Inh, threadsPerBlock>>>(E2I_state_ptr, E2I_PreIdx_ptr, E2I_PostIdx_ptr,E2I_pre_spike_ptr,E2I_post_input_ptr, E2I_post_V_ptr, numExc2Inh, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Exc, threadsPerBlock>>>(I2E_state_ptr, I2E_PreIdx_ptr, I2E_PostIdx_ptr,I2E_pre_spike_ptr,I2E_post_input_ptr, I2E_post_V_ptr, numInh2Exc, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Inh, threadsPerBlock>>>(I2I_state_ptr, I2I_PreIdx_ptr, I2I_PostIdx_ptr,I2I_pre_spike_ptr,I2I_post_input_ptr, I2I_post_V_ptr, numInh2Inh, dt);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_spike_times_exc, d_spike_times_exc, numExc * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_times_inh, d_spike_times_inh, numInh * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_exc, d_spike_counts_exc, numExc * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_inh, d_spike_counts_inh, numInh * sizeof(int), cudaMemcpyDeviceToHost);

    save_spike(h_spike_counts_exc,h_spike_times_exc,h_spike_counts_inh,h_spike_times_inh,numExc,numInh);
    
    // 释放内存
    // delete[] PopExc;
    // delete[] PopInh;
    // delete[] Exc2ExcSyn_AMPA;
    // delete[] Exc2InhSyn_AMPA;
    // delete[] Inh2ExcSyn_GABA;
    // delete[] Inh2InhSyn_GABA;
    delete[] h_spike_times_exc;
    delete[] h_spike_times_inh;
    delete[] h_spike_counts_exc;
    delete[] h_spike_counts_inh;

    // cudaFree(d_PopExc);
    // cudaFree(d_PopInh);
    // cudaFree(d_Exc2ExcSyn_AMPA);
    // cudaFree(d_Exc2InhSyn_AMPA);
    // cudaFree(d_Inh2ExcSyn_GABA);
    // cudaFree(d_Inh2InhSyn_GABA);
    cudaFree(d_spike_times_exc);
    cudaFree(d_spike_times_inh);
    cudaFree(d_spike_counts_exc);
    cudaFree(d_spike_counts_inh);
    return 0;
}