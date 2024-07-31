#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <vector>
#include <time.h>
#include <fstream>
using namespace std;

const int MAX_SPIKES = 10000; // 假设每个神经元最多记录1000次脉冲

struct LIFNeuron {
    float tau_m;
    float V_rest;
    float V_reset;
    float V_th;
    float R;
    float V;
    float tau_ref;
    float refractory_time;
    float input_current;
    bool spiked;
};

struct ExponentialSynapse {
    int pre;  // 突触前神经元索引
    int post; // 突触后神经元索引
    float g_max;
    float E_syn;
    float tau;
    float s;  // 突触的状态
};

__device__ void updateNeuron(LIFNeuron &neuron, float dt) {
    neuron.spiked = false;
    if (neuron.refractory_time > 0) {
        neuron.refractory_time -= dt;
        neuron.V = neuron.V_reset;
    } else {
        float V_inf = neuron.V_rest + neuron.R * neuron.input_current;
        neuron.V += dt * (V_inf - neuron.V) / neuron.tau_m;
        if (neuron.V >= neuron.V_th) {
            neuron.spiked = true;
            neuron.V = neuron.V_reset;
            neuron.refractory_time = neuron.tau_ref;
        }
    }
    neuron.input_current = 0;  // 重置输入电流为下一时间步准备
}

__device__ void updateSynapse(int idx, ExponentialSynapse &syn, LIFNeuron *preneurons, LIFNeuron *postneurons, float dt) {
    LIFNeuron *pre_neuron = &preneurons[syn.pre];
    LIFNeuron *post_neuron = &postneurons[syn.post];
    if (pre_neuron->spiked) {
        syn.s += 1.0;  // 突触前神经元发放动作电位，s增加
    }
    syn.s -= syn.s / syn.tau * dt;  // s的指数衰减
    float g_exp = syn.g_max * syn.s;
    float I_syn = g_exp * (syn.E_syn - post_neuron->V);  // 计算突触电流
    atomicAdd(&post_neuron->input_current, I_syn);  // 原子加，以避免并发写入问题
}

__global__ void simulateNeurons(LIFNeuron *neurons, int num_neurons, int input, float dt, float* spike_times, int* spike_counts, int time_step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_neurons) {
        int nid = tid;  // 当前神经元的索引
        atomicAdd(&neurons[nid].input_current, input);
        updateNeuron(neurons[nid], dt);
        if (neurons[nid].spiked) {
            int temp = atomicAdd(&spike_counts[nid], 1);
            spike_times[nid * MAX_SPIKES + temp] = time_step * dt;
        }
    }
}

__global__ void simulateSynapses(ExponentialSynapse *synapses, LIFNeuron *preneurons, LIFNeuron *postneurons, int num_synapses, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        updateSynapse(tid, synapses[tid], preneurons, postneurons, dt);
    }
}

int save_spike(const char* ESpikeFile,const char* ISpikeFile,int* h_spike_counts_exc, float* h_spike_times_exc, int* h_spike_counts_inh, float* h_spike_times_inh, int numExc, int numInh) {
    FILE* exc_spike_file = fopen(ESpikeFile, "w");
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < h_spike_counts_exc[i]; j++) {
            fprintf(exc_spike_file, "%f ", h_spike_times_exc[i * MAX_SPIKES + j]);
        }
        fprintf(exc_spike_file, "\n");
    }
    fclose(exc_spike_file);
    FILE* inh_spike_file = fopen(ISpikeFile, "w");
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
int main(int argc,char* argv[]) {
    const char* ESpikeFile=argv[1];
    const char* ISpikeFile=argv[2];
    int scale = 1;
    int numExc = 4096 * scale;
    int numInh = 1024 * scale;
    float connect_prob = 0.02;
    float dt = 0.1;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));

    //定义神经元群和突触连接
    LIFNeuron *PopExc = new LIFNeuron[numExc];
    LIFNeuron *d_PopExc;
    LIFNeuron *PopInh = new LIFNeuron[numInh];
    LIFNeuron *d_PopInh;

    ExponentialSynapse *Exc2ExcSyn_AMPA = new ExponentialSynapse[numExc * numExc];
    ExponentialSynapse *d_Exc2ExcSyn_AMPA;
    ExponentialSynapse *Exc2InhSyn_AMPA = new ExponentialSynapse[numExc * numInh];
    ExponentialSynapse *d_Exc2InhSyn_AMPA;
    ExponentialSynapse *Inh2ExcSyn_GABA = new ExponentialSynapse[numInh * numExc];
    ExponentialSynapse *d_Inh2ExcSyn_GABA;
    ExponentialSynapse *Inh2InhSyn_GABA = new ExponentialSynapse[numInh * numInh];
    ExponentialSynapse *d_Inh2InhSyn_GABA;

    // 初始化神经元参数
    for (int i = 0; i < numExc; i++) {
        PopExc[i].tau_m = 20;
        PopExc[i].V_rest = -60;
        PopExc[i].V_reset = -60;
        PopExc[i].V_th = -50;
        PopExc[i].R = 1;
        PopExc[i].V = -60;
        PopExc[i].tau_ref = 5;
        PopExc[i].refractory_time = 0;
        PopExc[i].input_current = 0;
        PopExc[i].spiked = false;
    }

    for (int i = 0; i < numInh; i++) {
        PopInh[i].tau_m = 20;
        PopInh[i].V_rest = -60;
        PopInh[i].V_reset = -60;
        PopInh[i].V_th = -50;
        PopInh[i].R = 1;
        PopInh[i].V = -60;
        PopInh[i].tau_ref = 5;
        PopInh[i].refractory_time = 0;
        PopInh[i].input_current = 0;
        PopInh[i].spiked = false;
    }

    // 分配和初始化突触连接参数
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                Exc2ExcSyn_AMPA[counter].pre = i;
                Exc2ExcSyn_AMPA[counter].post = j;
                Exc2ExcSyn_AMPA[counter].g_max = 0.3;
                Exc2ExcSyn_AMPA[counter].E_syn = 0.0;
                Exc2ExcSyn_AMPA[counter].tau = 5.0;
                Exc2ExcSyn_AMPA[counter].s = 0;
                counter++;
            }
        }
    }
    int numExc2Exc = counter;

    counter = 0;
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                Exc2InhSyn_AMPA[counter].pre = i;
                Exc2InhSyn_AMPA[counter].post = j;
                Exc2InhSyn_AMPA[counter].g_max = 0.3;
                Exc2InhSyn_AMPA[counter].E_syn = 0.0;
                Exc2InhSyn_AMPA[counter].tau = 5.0;
                Exc2InhSyn_AMPA[counter].s = 0;
                counter++;
            }
        }
    }
    int numExc2Inh = counter;

    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numExc; j++) {
            if (dis(generator) < connect_prob) {
                Inh2ExcSyn_GABA[counter].pre = i;
                Inh2ExcSyn_GABA[counter].post = j;
                Inh2ExcSyn_GABA[counter].g_max = 3.2;
                Inh2ExcSyn_GABA[counter].E_syn = -80;
                Inh2ExcSyn_GABA[counter].tau = 10.0;
                Inh2ExcSyn_GABA[counter].s = 0;
                counter++;
            }
        }
    }
    int numInh2Exc = counter;

    counter = 0;
    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numInh; j++) {
            if (dis(generator) < connect_prob) {
                Inh2InhSyn_GABA[counter].pre = i;
                Inh2InhSyn_GABA[counter].post = j;
                Inh2InhSyn_GABA[counter].g_max = 3.2;
                Inh2InhSyn_GABA[counter].E_syn = -80;
                Inh2InhSyn_GABA[counter].tau = 10.0;
                Inh2InhSyn_GABA[counter].s = 0;
                counter++;
            }
        }
    }
    int numInh2Inh = counter;

    // 初始化GPU变量
    cudaMalloc(&d_PopExc, numExc * sizeof(LIFNeuron));
    cudaMalloc(&d_PopInh, numInh * sizeof(LIFNeuron));
    cudaMemcpy(d_PopExc, PopExc, numExc * sizeof(LIFNeuron), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PopInh, PopInh, numInh * sizeof(LIFNeuron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Exc2ExcSyn_AMPA, numExc2Exc * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Exc2InhSyn_AMPA, numExc2Inh * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Inh2ExcSyn_GABA, numInh2Exc * sizeof(ExponentialSynapse));
    cudaMalloc(&d_Inh2InhSyn_GABA, numInh2Inh * sizeof(ExponentialSynapse));
    cudaMemcpy(d_Exc2ExcSyn_AMPA, Exc2ExcSyn_AMPA, numExc2Exc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Exc2InhSyn_AMPA, Exc2InhSyn_AMPA, numExc2Inh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inh2ExcSyn_GABA, Inh2ExcSyn_GABA, numInh2Exc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inh2InhSyn_GABA, Inh2InhSyn_GABA, numInh2Inh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice);

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
    int threadsPerBlock = 256;
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
    int steps = 10000;
    float input = 12.0;
    for (int t = 0; t < steps; t++) {
        // int input = (sin(2 * 3.1415 * t / freq) + 1) * 10;
        simulateNeurons<<<blocksPerGridExc, threadsPerBlock>>>(d_PopExc, numExc, input, dt, d_spike_times_exc, d_spike_counts_exc, t);
        simulateNeurons<<<blocksPerGridInh, threadsPerBlock>>>(d_PopInh, numInh, input, dt, d_spike_times_inh, d_spike_counts_inh, t);
        cudaDeviceSynchronize();

        simulateSynapses<<<blocksPerGridExc2Exc, threadsPerBlock>>>(d_Exc2ExcSyn_AMPA, d_PopExc, d_PopExc, numExc2Exc, dt);
        simulateSynapses<<<blocksPerGridExc2Inh, threadsPerBlock>>>(d_Exc2InhSyn_AMPA, d_PopExc, d_PopInh, numExc2Inh, dt);
        simulateSynapses<<<blocksPerGridInh2Exc, threadsPerBlock>>>(d_Inh2ExcSyn_GABA, d_PopInh, d_PopExc, numInh2Exc, dt);
        simulateSynapses<<<blocksPerGridInh2Inh, threadsPerBlock>>>(d_Inh2InhSyn_GABA, d_PopInh, d_PopInh, numInh2Inh, dt);
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

    // 打印结果
    // for (int i = 0; i < numExc; i++) {
    //     std::cout << "Excitatory neuron " << i << " spikes: " << h_spike_counts_exc[i] << std::endl;
    // }
    // for (int i = 0; i < numInh; i++) {
    //     std::cout << "Inhibitory neuron " << i << " spikes: " << h_spike_counts_inh[i] << std::endl;
    // }
    save_spike(ESpikeFile,ISpikeFile,h_spike_counts_exc,h_spike_times_exc,h_spike_counts_inh,h_spike_times_inh,numExc,numInh);
    
    // 释放内存
    delete[] PopExc;
    delete[] PopInh;
    delete[] Exc2ExcSyn_AMPA;
    delete[] Exc2InhSyn_AMPA;
    delete[] Inh2ExcSyn_GABA;
    delete[] Inh2InhSyn_GABA;
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