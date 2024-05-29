#include <iostream>
#include <cuda_runtime.h>
#include "/home/yangjinhao/enlarge-backup/enlarge-input-myself/src/third_party/cuda/helper_cuda.h"
#include <random>
const int MAX_SPIKES = 10000; // 假设每个神经元最多记录1000次脉冲
// struct LIFNeuron {
//     float *tau_m;
//     float *V_rest;
//     float *V_reset;
//     float *V_th;
//     float *R;
//     float V;
//     float tau_ref;
//     float refractory_time;
//     float input_current;
//     bool spiked;
// };

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
    // LIFNeuron* pre;
    // LIFNeuron* post;
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
    // LIFNeuron *pre_neuron = &neurons[syn.pre];
    // LIFNeuron *post_neuron = &neurons[syn.post];
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

__global__ void simulateNeurons(LIFNeuron *neurons, int num_neurons, int input, float dt, int fr, float* spike_times, int* spike_counts, int time_step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = 0; idx < num_neurons; idx +=blockDim.x*gridDim.x) {
        int nid = idx + tid;
        //neurons[nid].input_current += input;
        atomicAdd(&neurons[nid].input_current, input);
        updateNeuron(neurons[nid], dt);
        if(neurons[nid].spiked){
            fr += 1;
            int idx = atomicAdd(&spike_counts[tid], 1);
            spike_times[tid * MAX_SPIKES + idx] = time_step * dt;
        }
        __syncthreads();
    }
}

__global__ void simulateSynapses(ExponentialSynapse *synapses,LIFNeuron *preneurons, LIFNeuron *postneurons, int num_synapses, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;//当前thread的索引
    for (int idx = 0; idx < num_synapses; idx +=blockDim.x*gridDim.x) {
        int sid = idx + tid;
        updateSynapse(idx,synapses[sid], preneurons, postneurons, dt);
        __syncthreads();
    }
}
int save_spike(int* h_spike_counts_exc,float* h_spike_times_exc,int* h_spike_counts_inh,float* h_spike_times_inh,int numExc,int numInh){
    FILE* spike_file = fopen("spike_times.txt", "w");
    if (spike_file == NULL) {
        printf("ERROR: Open file spike_times.txt failed\n");
        return -1;
    }
    for (int i = 0; i < numExc; i++) {
        fprintf(spike_file, "Excitatory Neuron %d:\n", i);
        for (int j = 0; j < h_spike_counts_exc[i]; j++) {
            fprintf(spike_file, "%f ", h_spike_times_exc[i * MAX_SPIKES + j]);
        }
        fprintf(spike_file, "\n");
    }
    for (int i = 0; i < numInh; i++) {
        fprintf(spike_file, "Inhibitory Neuron %d:\n", i);
        for (int j = 0; j < h_spike_counts_inh[i]; j++) {
            fprintf(spike_file, "%f ", h_spike_times_inh[i * MAX_SPIKES + j]);
        }
        fprintf(spike_file, "\n");
    }
    fclose(spike_file);
    return 0;
}

// 主函数，设置和运行模拟
int main() {
    int scale = 1;
    int numExc = 4096*scale;
    int numInh = 1024*scale;
    float connect_prob = 0.02;
    float dt = 0.1;
    int freq = 20;
    std::default_random_engine generator;

    // std::string filename = "gpu.log";
    // FILE* file = fopen(filename.c_str(), "w+");
    // if (file == NULL) {
    //     printf("ERROR: Open file %s failed\n", filename.c_str());
    //     return -1;
    // }

    //定义神经元群和突触连接
    LIFNeuron *PopExc = new LIFNeuron[numExc];
    LIFNeuron *d_PopExc;
    LIFNeuron *PopInh = new LIFNeuron[numInh];
    LIFNeuron *d_PopInh;
    int fre = 0;
    int fri = 0;
    
    ExponentialSynapse *Exc2ExcSyn_AMPA = new ExponentialSynapse[numExc*numExc];
    ExponentialSynapse *d_Exc2ExcSyn_AMPA;
    ExponentialSynapse *Exc2InhSyn_AMPA = new ExponentialSynapse[numExc*numInh];
    ExponentialSynapse *d_Exc2InhSyn_AMPA;
    ExponentialSynapse *Inh2ExcSyn_GABA = new ExponentialSynapse[numInh*numExc];
    ExponentialSynapse *d_Inh2ExcSyn_GABA;
    ExponentialSynapse *Inh2InhSyn_GABA = new ExponentialSynapse[numInh*numInh];
    ExponentialSynapse *d_Inh2InhSyn_GABA;

    // 在GPU上分配内存
    checkCudaErrors(cudaMalloc(&d_PopExc, numExc * sizeof(LIFNeuron)));
    checkCudaErrors(cudaMalloc(&d_PopInh, numInh * sizeof(LIFNeuron)));
    checkCudaErrors(cudaMalloc(&d_Exc2ExcSyn_AMPA, numExc*numExc * sizeof(ExponentialSynapse)));
    checkCudaErrors(cudaMalloc(&d_Exc2InhSyn_AMPA, numExc*numInh * sizeof(ExponentialSynapse)));
    checkCudaErrors(cudaMalloc(&d_Inh2ExcSyn_GABA, numInh*numExc * sizeof(ExponentialSynapse)));
    checkCudaErrors(cudaMalloc(&d_Inh2InhSyn_GABA, numInh*numInh * sizeof(ExponentialSynapse)));
    
    //spike recorder
    float *d_spike_times_exc, *d_spike_times_inh;
    int *d_spike_counts_exc, *d_spike_counts_inh;
    float *h_spike_times_exc, *h_spike_times_inh;
    int *h_spike_counts_exc, *h_spike_counts_inh;

    h_spike_times_exc = new float[numExc * MAX_SPIKES];
    h_spike_times_inh = new float[numInh * MAX_SPIKES];
    h_spike_counts_exc = new int[numExc];
    h_spike_counts_inh = new int[numInh];
    memset(h_spike_counts_exc, 0, numExc * sizeof(int));
    memset(h_spike_counts_inh, 0, numInh * sizeof(int));

    cudaMalloc(&d_spike_times_exc, numExc * MAX_SPIKES * sizeof(float));
    cudaMalloc(&d_spike_times_inh, numInh * MAX_SPIKES * sizeof(float));
    cudaMalloc(&d_spike_counts_exc, numExc * sizeof(int));
    cudaMalloc(&d_spike_counts_inh, numInh * sizeof(int));

    cudaMemcpy(d_spike_counts_exc, h_spike_counts_exc, numExc * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spike_counts_inh, h_spike_counts_inh, numInh * sizeof(int), cudaMemcpyHostToDevice);

    // 初始化神经元和突触
    for (int i = 0; i < numExc; i++) {
        PopExc[i].tau_m = 7.5;
        PopExc[i].V_rest = -70.0;
        PopExc[i].V_reset = -75.0;
        PopExc[i].V_th = -50.0;
        PopExc[i].R = 0.05;
        PopExc[i].V = -60.0;
        PopExc[i].tau_ref = 2.0;
        PopExc[i].refractory_time = 0.0;
        PopExc[i].input_current = 0.0;
        PopExc[i].spiked = false;
    }

    for (int i = 0; i < numInh; i++) {
        PopInh[i].tau_m = 7.5;
        PopInh[i].V_rest = -70.0;
        PopInh[i].V_reset = -75.0;
        PopInh[i].V_th = -50.0;
        PopInh[i].R = 0.05;
        PopInh[i].V = -60;
        PopInh[i].tau_ref = 2.0;
        PopInh[i].refractory_time = 0.0;
        PopInh[i].input_current = 0.0;
        PopInh[i].spiked = false;
    }
    
    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numExc; j++) {
            Exc2ExcSyn_AMPA[i*numExc+j].pre = i;
            Exc2ExcSyn_AMPA[i*numExc+j].post = j;
            if ((float)(rand() % 10000) / 10000 < connect_prob) {
                Exc2ExcSyn_AMPA[i*numExc+j].g_max = 0.3;
            }
            else{
                Exc2ExcSyn_AMPA[i*numExc+j].g_max = 0;
            }
            Exc2ExcSyn_AMPA[i*numExc+j].E_syn = 0.0;
            Exc2ExcSyn_AMPA[i*numExc+j].tau = 5.0;
            Exc2ExcSyn_AMPA[i*numExc+j].s = 0.0;
        }
    }

    for (int i = 0; i < numExc; i++) {
        for (int j = 0; j < numInh; j++) {
            Exc2InhSyn_AMPA[i*numInh+j].pre = i;
            Exc2InhSyn_AMPA[i*numInh+j].post = j;
            if ((float)(rand() % 10000) / 10000 < connect_prob) {
                Exc2InhSyn_AMPA[i*numInh+j].g_max = 0.3;
            }
            else{
                Exc2InhSyn_AMPA[i*numInh+j].g_max = 0;
            }
            Exc2InhSyn_AMPA[i*numInh+j].E_syn = 0.0;
            Exc2InhSyn_AMPA[i*numInh+j].tau = 5.0;
            Exc2InhSyn_AMPA[i*numInh+j].s = 0.0;
        }
    }

    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numExc; j++) {
            Inh2ExcSyn_GABA[i*numExc+j].pre = i;
            Inh2ExcSyn_GABA[i*numExc+j].post = j;
            if ((float)(rand() % 10000) / 10000 < connect_prob) {
                Inh2ExcSyn_GABA[i*numExc+j].g_max = 3.2;
            }
            else{
                Inh2ExcSyn_GABA[i*numExc+j].g_max = 0;
            }
            Inh2ExcSyn_GABA[i*numExc+j].E_syn = -80.0;
            Inh2ExcSyn_GABA[i*numExc+j].tau = 10.0;
            Inh2ExcSyn_GABA[i*numExc+j].s = 0.0;
        }
    }

    for (int i = 0; i < numInh; i++) {
        for (int j = 0; j < numInh; j++) {
            Inh2InhSyn_GABA[i*numInh+j].pre = i;
            Inh2InhSyn_GABA[i*numInh+j].post = j;
            if ((float)(rand() % 10000) / 10000 < connect_prob) {
                Inh2InhSyn_GABA[i*numInh+j].g_max = 3.2;
            }
            else{
                Inh2InhSyn_GABA[i*numInh+j].g_max = 0;
            }
            Inh2InhSyn_GABA[i*numInh+j].E_syn = -80.0;
            Inh2InhSyn_GABA[i*numInh+j].tau = 10.0;
            Inh2InhSyn_GABA[i*numInh+j].s = 0.0;
        }
    }

    // 将初始化的神经元突触拷贝到GPU上
    checkCudaErrors(cudaMemcpy(d_PopExc,PopExc, numExc * sizeof(LIFNeuron), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_PopInh,PopInh, numInh * sizeof(LIFNeuron), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Exc2ExcSyn_AMPA,Exc2ExcSyn_AMPA, numExc*numExc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Exc2InhSyn_AMPA,Exc2InhSyn_AMPA, numExc*numInh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Inh2ExcSyn_GABA,Inh2ExcSyn_GABA, numInh*numExc * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Inh2InhSyn_GABA,Inh2InhSyn_GABA, numInh*numInh * sizeof(ExponentialSynapse), cudaMemcpyHostToDevice));

    //记录运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    int threadsPerBlock = 256;
    // 运行模拟
    for (int i = 0; i < 10000; i++){
        std::poisson_distribution<int> pd(freq);
        float I_pos = pd(generator);
        // std::cout << I_pos << "\n";

        //突触更新，block分配synapse数量/256
        int synE2EblocksPerGrid = (numExc*numExc + threadsPerBlock - 1) / threadsPerBlock;
        simulateSynapses<<<synE2EblocksPerGrid, threadsPerBlock>>>(d_Exc2ExcSyn_AMPA, d_PopExc, d_PopExc, numExc*numExc, dt);
        int synE2IblocksPerGrid = (numExc*numInh + threadsPerBlock - 1) / threadsPerBlock;
        simulateSynapses<<<synE2IblocksPerGrid, threadsPerBlock>>>(d_Exc2InhSyn_AMPA, d_PopExc, d_PopInh, numExc*numInh, dt);
        int synI2EblocksPerGrid = (numInh*numExc + threadsPerBlock - 1) / threadsPerBlock;
        simulateSynapses<<<synI2EblocksPerGrid, threadsPerBlock>>>(d_Inh2ExcSyn_GABA, d_PopInh, d_PopExc, numInh*numExc, dt);
        int synI2IblocksPerGrid = (numInh*numInh + threadsPerBlock - 1) / threadsPerBlock;
        simulateSynapses<<<synI2IblocksPerGrid, threadsPerBlock>>>(d_Inh2InhSyn_GABA, d_PopInh, d_PopInh, numInh*numInh, dt);

        //神经元更新
        int ExcblocksPerGrid = (numExc + threadsPerBlock - 1) / threadsPerBlock;
        simulateNeurons<<<ExcblocksPerGrid, threadsPerBlock>>>(d_PopExc, numExc, I_pos, dt, fre, d_spike_times_exc, d_spike_counts_exc, i);
        int InhblocksPerGrid = (numInh + threadsPerBlock - 1) / threadsPerBlock;
        simulateNeurons<<<InhblocksPerGrid, threadsPerBlock>>>(d_PopInh, numInh, I_pos, dt, fri, d_spike_times_inh, d_spike_counts_inh, i);

        // checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // fprintf(file, "%d ", fre);
        // fprintf(file, "%d ", fri);
        // fprintf(file, "\n");
    }
    checkCudaErrors(cudaDeviceSynchronize());

    //统计时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // save_spikes
    cudaMemcpy(h_spike_times_exc, d_spike_times_exc, numExc * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_times_inh, d_spike_times_inh, numInh * MAX_SPIKES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_exc, d_spike_counts_exc, numExc * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spike_counts_inh, d_spike_counts_inh, numInh * sizeof(int), cudaMemcpyDeviceToHost);
    save_spike(h_spike_counts_exc,h_spike_times_exc,h_spike_counts_inh,h_spike_times_inh,numExc,numInh);

    // 清理资源
    cudaFree(d_PopExc);
    cudaFree(d_PopInh);
    cudaFree(d_Exc2ExcSyn_AMPA);
    cudaFree(d_Exc2InhSyn_AMPA);
    cudaFree(d_Inh2ExcSyn_GABA);
    cudaFree(d_Inh2InhSyn_GABA);
    cudaFree(d_spike_times_exc);
    cudaFree(d_spike_counts_exc);
    cudaFree(d_spike_times_inh);
    cudaFree(d_spike_counts_inh);
    delete[] PopExc;
    delete[] PopInh;
    delete[] Exc2ExcSyn_AMPA;
    delete[] Exc2InhSyn_AMPA;
    delete[] Inh2ExcSyn_GABA;
    delete[] Inh2InhSyn_GABA;

    return 0;
}