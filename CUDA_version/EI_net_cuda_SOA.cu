#include <iostream>
#include <cuda_runtime.h>
#include "/home/yangjinhao/enlarge-backup/enlarge-input-myself/src/third_party/cuda/helper_cuda.h"
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

__global__ void simulateNeuronsFixpara(LIFNeuron *neurons, int num_neurons, int input, float dt, float* spike_times, int* spike_counts, int time_step){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<num_neurons){
        atomicAdd(&neurons[tid].input_current,input);
        neurons[tid].spiked = false;
        if (neurons[tid].refractory_time > 0) {
            neurons[tid].refractory_time -= dt;
            neurons[tid].V = -60.0;//reset voltage
        } else {
            //V_inf=E_L+RI;
            //V=V+dt*(V_inf-V)/tau_m
            //ouler
            float V_inf = -60.0 + 1.0 * neurons[tid].input_current;//EL
            neurons[tid].V += dt * (V_inf - neurons[tid].V) / 20.0;//tau_m
            if (neurons[tid].V >= -50.0) {//V_th
                neurons[tid].spiked = true;
                neurons[tid].V = -60.0;//V_reset
                neurons[tid].refractory_time = 5.0;//tau_ref
            }
        }
        neurons[tid].input_current = 0;  // 重置输入电流为下一时间步准备
        if (neurons[tid].spiked) {
            int temp = atomicAdd(&spike_counts[tid], 1);
            spike_times[tid * MAX_SPIKES + temp] = time_step * dt;
        }
    }
}

__global__ void simulateSynapsesFixparaAmpa(ExponentialSynapse *synapses, LIFNeuron *preneurons, LIFNeuron *postneurons, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        LIFNeuron *pre_neuron = &preneurons[synapses[tid].pre];
        LIFNeuron *post_neuron = &postneurons[synapses[tid].post];
        if (pre_neuron->spiked) {
            synapses[tid].s += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        synapses[tid].s -= synapses[tid].s / 5.0 * dt;  // s的指数衰减,tau
        float g_exp = 0.3 * synapses[tid].s;//g_max
        float I_syn = g_exp * (0.0 - post_neuron->V);  // 计算突触电流,EL
        atomicAdd(&post_neuron->input_current, I_syn);  // 原子加，以避免并发写入问题
    }
}

__global__ void simulateSynapsesFixparaGaba(ExponentialSynapse *synapses, LIFNeuron *preneurons, LIFNeuron *postneurons, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        LIFNeuron *pre_neuron = &preneurons[synapses[tid].pre];
        LIFNeuron *post_neuron = &postneurons[synapses[tid].post];
        if (pre_neuron->spiked) {
            synapses[tid].s += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        synapses[tid].s -= synapses[tid].s / 10.0 * dt;  // s的指数衰减,tau
        float g_exp = 3.2 * synapses[tid].s;//g_max
        float I_syn = g_exp * (-80 - post_neuron->V);  // 计算突触电流,EL
        atomicAdd(&post_neuron->input_current, I_syn);  // 原子加，以避免并发写入问题
    }
}

__global__ void InitSynapse(){

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
    std::default_random_engine generator;
    //定义神经元群和突触连接
    struct LIFNeuron *PopExc;
    struct LIFNeuron *d_PopExc;
    struct LIFNeuron *PopInh;
    struct LIFNeuron *d_PopInh;

    // 初始化神经元参数
    auto start_init_neuron = high_resolution_clock::now();
    PopExc->V.resize(numExc);
    PopExc->refractory_time = vector<float>(numExc,0.0);
    PopExc->input_current = vector<float>(numExc,0.0);
    PopExc->spiked = vector<bool>(numExc,false);

    PopInh->V = vector<float>(numInh,-60.0);
    PopInh->refractory_time = vector<float>(numInh,0.0);
    PopInh->input_current = vector<float>(numInh,0.0);
    PopInh->spiked = vector<bool>(numInh,false);

    //

    auto end_init_neuron = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init_neuron - start_init_neuron);
    cout << "neurons initTime: " << duration.count() << " ms" << endl;

    vector<ExponentialSynapse> Exc2ExcSyn_AMPA;
    ExponentialSynapse *d_Exc2ExcSyn_AMPA;
    vector<ExponentialSynapse> Exc2InhSyn_AMPA;
    ExponentialSynapse *d_Exc2InhSyn_AMPA;
    vector<ExponentialSynapse> Inh2ExcSyn_GABA;
    ExponentialSynapse *d_Inh2ExcSyn_GABA;
    vector<ExponentialSynapse> Inh2InhSyn_GABA;
    ExponentialSynapse *d_Inh2InhSyn_GABA;

    auto start_init = high_resolution_clock::now();

    // 分配和初始化突触连接参数
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                ExponentialSynapse syn;
                syn.pre = i;
                syn.post = j;
                syn.s = 0;
                Exc2ExcSyn_AMPA.push_back(syn);
                counter++;
            }
        }
    }
    int numExc2Exc = counter;

    counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                ExponentialSynapse syn;
                syn.pre = i;
                syn.post = j;
                syn.s = 0;
                Exc2InhSyn_AMPA.push_back(syn);
                counter++;
            }
        }
    }
    int numExc2Inh = counter;

    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                ExponentialSynapse syn;
                syn.pre = i;
                syn.post = j;
                syn.s = 0;
                Inh2ExcSyn_GABA.push_back(syn);
                counter++;
            }
        }
    }
    int numInh2Exc = counter;

    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                ExponentialSynapse syn;
                syn.pre = i;
                syn.post = j;
                syn.s = 0;
                Inh2InhSyn_GABA.push_back(syn);
                counter++;
            }
        }
    }
    int numInh2Inh = counter;

    auto end_init = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init - start_init);
    cout << "initTime: " << duration.count() << " ms" << endl;

    // 初始化GPU变量
    cudaMalloc(&d_PopExc, numExc * sizeof(LIFNeuron));
    cudaMalloc(&d_PopInh, numInh * sizeof(LIFNeuron));
    cudaMemcpy(d_PopExc, PopExc, numExc * sizeof(LIFNeuron), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PopInh, PopInh, numInh * sizeof(LIFNeuron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Exc2ExcSyn_AMPA, numExc2Exc * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Exc2InhSyn_AMPA, numExc2Inh * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Inh2ExcSyn_GABA, numInh2Exc * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Inh2InhSyn_GABA, numInh2Inh * sizeof(ExponentialSynapse));
    cudaMemcpy(d_Exc2ExcSyn_AMPA, Exc2ExcSyn_AMPA.data(), numExc2Exc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Exc2InhSyn_AMPA, Exc2InhSyn_AMPA.data(), numExc2Inh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inh2ExcSyn_GABA, Inh2ExcSyn_GABA.data(), numInh2Exc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inh2InhSyn_GABA, Inh2InhSyn_GABA.data(), numInh2Inh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);

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
        simulateNeuronsFixpara<<<blocksPerGridExc, threadsPerBlock>>>(d_PopExc, numExc, input, dt, d_spike_times_exc, d_spike_counts_exc, t);
        simulateNeuronsFixpara<<<blocksPerGridInh, threadsPerBlock>>>(d_PopInh, numInh, input, dt, d_spike_times_inh, d_spike_counts_inh, t);
        cudaDeviceSynchronize();

        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Exc, threadsPerBlock>>>(d_Exc2ExcSyn_AMPA, d_PopExc, d_PopExc, numExc2Exc, dt);
        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Inh, threadsPerBlock>>>(d_Exc2InhSyn_AMPA, d_PopExc, d_PopInh, numExc2Inh, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Exc, threadsPerBlock>>>(d_Inh2ExcSyn_GABA, d_PopInh, d_PopExc, numInh2Exc, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Inh, threadsPerBlock>>>(d_Inh2InhSyn_GABA, d_PopInh, d_PopInh, numInh2Inh, dt);
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
    delete[] PopExc;
    delete[] PopInh;
    // delete[] Exc2ExcSyn_AMPA;
    // delete[] Exc2InhSyn_AMPA;
    // delete[] Inh2ExcSyn_GABA;
    // delete[] Inh2InhSyn_GABA;
    delete[] h_spike_times_exc;
    delete[] h_spike_times_inh;
    delete[] h_spike_counts_exc;
    delete[] h_spike_counts_inh;

    cudaFree(d_PopExc);
    cudaFree(d_PopInh);
    cudaFree(d_Exc2ExcSyn_AMPA);
    cudaFree(d_Exc2InhSyn_AMPA);
    cudaFree(d_Inh2ExcSyn_GABA);
    cudaFree(d_Inh2InhSyn_GABA);
    cudaFree(d_spike_times_exc);
    cudaFree(d_spike_times_inh);
    cudaFree(d_spike_counts_exc);
    cudaFree(d_spike_counts_inh);
    return 0;
}