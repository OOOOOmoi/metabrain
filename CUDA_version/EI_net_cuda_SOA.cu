#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <vector>
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include "/home/yangjinhao/enlarge-backup/enlarge-develop/src/third_party/cuda/helper_cuda.h"
#include "/home/yangjinhao/CUDA_Freshman-master/include/freshman.h"
using namespace std;
using namespace std::chrono;

const int MAX_SPIKES = 10000; // 假设每个神经元最多记录1000次脉冲

struct LIFData
{
    float V_init=-60.0;
    float Ref=0.0;
    float I_init=0.0;
    int SpikeState=0;
};
struct LIFNeuron {
    float* V;
    float* refractory_time;
    float* input_current;
    int* spiked;
};

struct ExponentialSynapse {
    int* pre;  // 突触前神经元索引
    int* post; // 突触后神经元索引
    float* s;  // 突触的状态
};

struct ExponentialSynapseHOST {
    vector<int> pre;
    vector<int> post;
    vector<float> s;
};


void initNeurons(LIFNeuron* data,int num,LIFData para){
    data->V=(float*)malloc(num*sizeof(float));
    data->refractory_time=(float*)malloc(num*sizeof(float));
    data->input_current=(float*)malloc(num*sizeof(float));
    data->spiked=(int*)malloc(num*sizeof(float));
    for (int i = 0; i < num; i++)
    {
        data->V[i]=para.V_init;
        data->refractory_time[i]=para.Ref;
        data->spiked[i]=para.SpikeState;
        data->input_current[i]=para.I_init;
    }
}

void* cudaAllocNeurons(LIFNeuron* data, int num){
    LIFNeuron* D_data=NULL;
    checkCudaErrors(cudaMalloc(&D_data,sizeof(LIFNeuron)));
    checkCudaErrors(cudaMemset(D_data,0,sizeof(LIFNeuron)));
    LIFNeuron* d_data=(LIFNeuron*)malloc(sizeof(LIFNeuron));
    memset(d_data,0,sizeof(LIFNeuron));

    checkCudaErrors(cudaMalloc(&d_data->V,sizeof(float)*num));
    checkCudaErrors(cudaMemcpy(d_data->V,&data->V,sizeof(float)*num,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_data->refractory_time,sizeof(float)*num));
    checkCudaErrors(cudaMemcpy(d_data->refractory_time,&data->refractory_time,sizeof(float)*num,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_data->input_current,sizeof(float)*num));
    checkCudaErrors(cudaMemcpy(d_data->input_current,&data->input_current,sizeof(float)*num,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_data->spiked,sizeof(float)*num));
    checkCudaErrors(cudaMemcpy(d_data->spiked,&data->spiked,sizeof(float)*num,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_data,d_data,sizeof(LIFNeuron),cudaMemcpyHostToDevice));
    free(d_data);
    return D_data;
}

void FreeNeuron(LIFNeuron* data){
    // 释放 initNeurons 分配的内存
    free(data->V);
    free(data->refractory_time);
    free(data->input_current);
    free(data->spiked);
}

void cudaFreeNeurons(LIFNeuron* d_data) {
    cudaFree(d_data->V);
    cudaFree(d_data->refractory_time);
    cudaFree(d_data->input_current);
    cudaFree(d_data->spiked);
    cudaFree(d_data); // 释放 d_data 本身
}

int initSynapse(ExponentialSynapseHOST* data,int numPre,int numPost,float connect_prob){
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int counter = 0;
    for (int i = 0; i < numPre; i++) {
        for (int j = 0; j < numPost; j++) {
            if (dis(generator) < connect_prob) {
                data->pre.push_back(i);
                data->post.push_back(j);
                data->s.push_back(0.0);
                counter++;
            }
        }
    }
    return counter;
}

void* cudaAllocSynapse(ExponentialSynapseHOST* data,int num){
    ExponentialSynapse* D_data=NULL;
    checkCudaErrors(cudaMalloc(&D_data,sizeof(ExponentialSynapse)));
    checkCudaErrors(cudaMemset(D_data,0,sizeof(ExponentialSynapse)));
    ExponentialSynapse* d_data=(ExponentialSynapse*)malloc(sizeof(ExponentialSynapse));
    memset(d_data,0,sizeof(ExponentialSynapse));
    
    cudaMalloc(&d_data->pre,sizeof(int)*num);
    cudaMemcpy(d_data->pre,data->pre.data(),sizeof(int)*num,cudaMemcpyHostToDevice);
    cudaMalloc(&d_data->post,sizeof(int)*num);
    cudaMemcpy(d_data->post,data->post.data(),sizeof(int)*num,cudaMemcpyHostToDevice);
    cudaMalloc(&d_data->s,sizeof(float)*num);
    cudaMemcpy(d_data->s,data->s.data(),sizeof(float)*num,cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemcpy(D_data,d_data,sizeof(ExponentialSynapse),cudaMemcpyHostToDevice));
    free(d_data);
    return D_data;
}

void cudaFreeSynapse(ExponentialSynapse* d_data) {
    cudaFree(d_data->pre);
    cudaFree(d_data->post);
    cudaFree(d_data->s);
    cudaFree(d_data); // 释放 d_data 本身
}

__global__ void simulateNeuronsFixpara(LIFNeuron* group, int num_neurons, int input, float dt, float* spike_times, int* spike_counts, int time_step){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    // if(tid==0){
    //     printf("Time: %f",time_step*dt);
    // }
    if(tid<num_neurons){
        group->input_current[tid]+=input;
        group->spiked[tid] = 0;
        if (group->refractory_time[tid] > 0) {
            group->refractory_time[tid] -= dt;
            group->V[tid] = -60.0;//reset voltage
        } else {
            //V_inf=E_L+RI;
            //V=V+dt*(V_inf-V)/tau_m
            //ouler
            float V_inf = -60.0 + 1.0 * group->input_current[tid];//EL
            group->V[tid] += dt * (V_inf - group->V[tid]) / 20.0;//tau_m
            if (group->V[tid] >= -50.0) {//V_th
                group->spiked[tid] = 1;
                group->V[tid] = -60.0;//V_reset
                group->refractory_time[tid] = 5.0;//tau_ref
            }
        }
        group->input_current[tid] = 0;  // 重置输入电流为下一时间步准备
        if (group->spiked[tid]) {
            int temp = atomicAdd(&spike_counts[tid], 1);
            spike_times[tid * MAX_SPIKES + temp] = time_step * dt;
        }
    }
}

__global__ void simulateSynapsesFixparaAmpa(ExponentialSynapse* syn,LIFNeuron* PreGroup,LIFNeuron* PostGroup, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        if (PreGroup->spiked[syn->pre[tid]]) {
            syn->s[tid] += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        syn->s[tid] -= syn->s[tid] / 5.0 * dt;  // s的指数衰减,tau
        float g_exp = 0.3 * syn->s[tid];//g_max
        float I_syn = g_exp * (0.0 - PostGroup->V[syn->post[tid]]);  // 计算突触电流,EL
        atomicAdd(&PostGroup->input_current[syn->post[tid]], I_syn);  // 原子加，以避免并发写入问题
    }
}

__global__ void simulateSynapsesFixparaGaba(ExponentialSynapse* syn,LIFNeuron* PreGroup,LIFNeuron* PostGroup, int num_synapses, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前thread的索引
    if (tid < num_synapses) {
        if (PreGroup->spiked[syn->pre[tid]]) {
            syn->s[tid] += 1.0;  // 突触前神经元发放动作电位，s增加
        }
        syn->s[tid] -= syn->s[tid] / 10.0 * dt;  // s的指数衰减,tau
        float g_exp = 3.2 * syn->s[tid];//g_max
        float I_syn = g_exp * (-80.0 - PostGroup->V[syn->post[tid]]);  // 计算突触电流,EL
        atomicAdd(&PostGroup->input_current[syn->post[tid]], I_syn);  // 原子加，以避免并发写入问题
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
    cout<<"spike of GroupExc record over!"<<endl;
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
    //定义神经元群和突触连接
    LIFData LIFPara;
    LIFNeuron *PopExc=new LIFNeuron;
    LIFNeuron *PopInh=new LIFNeuron;
    initNeurons(PopExc,numExc,LIFPara);
    initNeurons(PopInh,numInh,LIFPara);
    LIFNeuron* d_PopExc=(LIFNeuron*)cudaAllocNeurons(PopExc,numExc);
    LIFNeuron* d_PopInh=(LIFNeuron*)cudaAllocNeurons(PopInh,numInh);

    // 初始化神经元参数
    auto start_init_neuron = high_resolution_clock::now();
    

    auto end_init_neuron = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_init_neuron - start_init_neuron);
    cout << "neurons initTime: " << duration.count() << " ms" << endl;

    ExponentialSynapseHOST* Exc2ExcSyn_AMPA=new ExponentialSynapseHOST;
    ExponentialSynapse* d_Exc2ExcSyn_AMPA;
    ExponentialSynapseHOST* Exc2InhSyn_AMPA=new ExponentialSynapseHOST;
    ExponentialSynapse* d_Exc2InhSyn_AMPA;
    ExponentialSynapseHOST* Inh2ExcSyn_GABA=new ExponentialSynapseHOST;
    ExponentialSynapse* d_Inh2ExcSyn_GABA;
    ExponentialSynapseHOST* Inh2InhSyn_GABA=new ExponentialSynapseHOST;
    ExponentialSynapse* d_Inh2InhSyn_GABA;

    auto start_init = high_resolution_clock::now();

    // 分配和初始化突触连接参数
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    int numExc2Exc = initSynapse(Exc2ExcSyn_AMPA,numExc,numExc,connect_prob);
    int numExc2Inh = initSynapse(Exc2InhSyn_AMPA,numExc,numInh,connect_prob);
    int numInh2Exc = initSynapse(Inh2ExcSyn_GABA,numInh,numExc,connect_prob);
    int numInh2Inh = initSynapse(Inh2InhSyn_GABA,numInh,numInh,connect_prob);

    d_Exc2ExcSyn_AMPA=(ExponentialSynapse*)cudaAllocSynapse(Exc2ExcSyn_AMPA,numExc2Exc);
    d_Exc2InhSyn_AMPA=(ExponentialSynapse*)cudaAllocSynapse(Exc2InhSyn_AMPA,numExc2Inh);
    d_Inh2ExcSyn_GABA=(ExponentialSynapse*)cudaAllocSynapse(Inh2ExcSyn_GABA,numInh2Exc);
    d_Inh2InhSyn_GABA=(ExponentialSynapse*)cudaAllocSynapse(Inh2InhSyn_GABA,numInh2Inh);

    auto end_init = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_init - start_init);
    cout << "Synapses initTime: " << duration.count() << " ms" << endl;

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
    double start=cpuSecond();
    float input = 12.0;
    for (int t = 0; t < steps; t++) {
        simulateNeuronsFixpara<<<blocksPerGridExc, threadsPerBlock>>>(d_PopExc, numExc, input, dt, d_spike_times_exc, d_spike_counts_exc, t);
        simulateNeuronsFixpara<<<blocksPerGridInh, threadsPerBlock>>>(d_PopInh, numInh, input, dt, d_spike_times_inh, d_spike_counts_inh, t);
        cudaDeviceSynchronize();

        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Exc, threadsPerBlock>>>(d_Exc2ExcSyn_AMPA,d_PopExc,d_PopExc, numExc2Exc, dt);
        simulateSynapsesFixparaAmpa<<<blocksPerGridExc2Inh, threadsPerBlock>>>(d_Exc2InhSyn_AMPA,d_PopExc,d_PopInh, numExc2Inh, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Exc, threadsPerBlock>>>(d_Inh2ExcSyn_GABA,d_PopInh,d_PopExc, numInh2Exc, dt);
        simulateSynapsesFixparaGaba<<<blocksPerGridInh2Inh, threadsPerBlock>>>(d_Inh2InhSyn_GABA,d_PopInh,d_PopInh, numInh2Inh, dt);
        cudaDeviceSynchronize();
    }
    double isElaps=cpuSecond()-start;
    printf("Time of Simulation: %fms\n",1000*isElaps);


    cudaMemcpy(h_spike_times_exc, d_spike_times_exc, numExc * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_times_inh, d_spike_times_inh, numInh * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_exc, d_spike_counts_exc, numExc * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_inh, d_spike_counts_inh, numInh * sizeof(int), cudaMemcpyDeviceToHost);

    save_spike(h_spike_counts_exc,h_spike_times_exc,h_spike_counts_inh,h_spike_times_inh,numExc,numInh);
    
    // 释放内存
    FreeNeuron(PopExc);
    FreeNeuron(PopInh);
    cudaFreeNeurons(d_PopExc);
    cudaFreeNeurons(d_PopInh);
    delete Exc2ExcSyn_AMPA;
    delete Exc2InhSyn_AMPA;
    delete Inh2ExcSyn_GABA;
    delete Inh2InhSyn_GABA;
    cudaFreeSynapse(d_Exc2ExcSyn_AMPA);
    cudaFreeSynapse(d_Exc2InhSyn_AMPA);
    cudaFreeSynapse(d_Inh2ExcSyn_GABA);
    cudaFreeSynapse(d_Inh2InhSyn_GABA);
    delete[] h_spike_times_exc;
    delete[] h_spike_times_inh;
    delete[] h_spike_counts_exc;
    delete[] h_spike_counts_inh;
    cudaFree(d_spike_times_exc);
    cudaFree(d_spike_times_inh);
    cudaFree(d_spike_counts_exc);
    cudaFree(d_spike_counts_inh);
    return 0;
}