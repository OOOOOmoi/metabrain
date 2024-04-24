#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "src.h"
#include <time.h>
#include <fstream>
#include <pthread.h>
#include <threadpool.hpp>
using namespace std;
using namespace std::chrono;

const string E_spike_file = "E_spike_times.txt";
const string I_spike_file = "I_spike_times.txt";

class EInet{
    public:
    float V_init=-70.0;
    float input=12.0;
    LifParams lifpara={-60.0,-60.0,-50.0,20.0,5.0,1.0};
    //revers,rest,thresold,tau,ref,R
    float g_AMPA=0.3;
    float tau_AMPA=5.0;
    float E_AMPA=0.0;
    float g_GABA=3.2;
    float tau_GABA=10.0;
    float E_GABA=-80.0;
    float dt=0.1;
    float fixprob=0.02;
    int numE=20000;
    int numI=5000;
    
    vector<LIFNeuron> groupE;
    vector<LIFNeuron> groupI;
    vector<ExponentialSynapse> E2E;
    vector<ExponentialSynapse> E2I;
    vector<ExponentialSynapse> I2I;
    vector<ExponentialSynapse> I2E;
    vector<vector<float>> spikesE; // 存储E神经元的spike时间
    vector<vector<float>> spikesI; // 存储I神经元的spike时间



    EInet(){
        createneuron();
        createsynapse();
        spikesE.resize(numE);
        spikesI.resize(numI);
    }


    void createneuron(){
        for (int i = 0; i < numE; i++){
            groupE.emplace_back(LIFNeuron(lifpara.tau_m,lifpara.V_rest,
                                lifpara.V_reset,lifpara.V_th,lifpara.R,lifpara.t_refractory,-70.0));
        }
        for (int i = 0; i < numI; i++){
            groupI.emplace_back(LIFNeuron(lifpara.tau_m,lifpara.V_rest,
                                lifpara.V_reset,lifpara.V_th,lifpara.R,lifpara.t_refractory,-70.0));
        }
        
    }
    void createsynapse(){
        for (auto& preneuron : groupE){
            for (auto& postneuron : groupE){
                double temp=(rand() % 100) * 0.01;
                if (temp<fixprob){
                    E2E.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_AMPA,E_AMPA,tau_AMPA));
                }
            }
        }
        for (auto& preneuron : groupE){
            for (auto& postneuron : groupI){
                double temp=(rand() % 100) * 0.01;
                if (temp<fixprob){
                    E2I.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_AMPA,E_AMPA,tau_AMPA));
                }
            }
        }
        for (auto& preneuron : groupI){
            for (auto& postneuron : groupE){
                double temp=(rand() % 100) * 0.01;
                if (temp<fixprob){
                    I2E.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_GABA,E_GABA,tau_GABA));
                }
            }
        }
        for (auto& preneuron : groupI){
            for (auto& postneuron : groupI){
                double temp=(rand() % 100) * 0.01;
                if (temp<fixprob){
                    I2I.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_GABA,E_GABA,tau_GABA));
                }
            }
        }
    }
    void saveSpikeTimes() {
        ofstream E_spike_out(E_spike_file);
        ofstream I_spike_out(I_spike_file);
        if (E_spike_out.is_open() && I_spike_out.is_open()) {
            for (const auto& spikes : spikesE) {
                for (const auto& time : spikes) {
                    E_spike_out << time << " ";
                }
                E_spike_out << endl;
            }
            for (const auto& spikes : spikesI) {
                for (const auto& time : spikes) {
                    I_spike_out << time << " ";
                }
                I_spike_out << endl;
            }
            E_spike_out.close();
            I_spike_out.close();
            cout << "Spike times saved to files." << endl;
        } else {
            cout << "Error: Unable to open spike time files." << endl;
        }
    }
};


struct ThreadData {
    vector<LIFNeuron>* neurons;
    vector<vector<float>>* spikes;
    float dt;
    float currentTime;
};

void updateNeurons(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    vector<LIFNeuron>* neurons = data->neurons;
    vector<vector<float>>* spikes = data->spikes;
    float dt = data -> dt;
    float currentTime=data->currentTime;
    for (int j = 0; j < neurons->size(); ++j) {
        (*neurons)[j].receiveCurrent(12.0);
        (*neurons)[j].update(dt);
        if ((*neurons)[j].hasFired()) {
            (*spikes)[j].push_back(currentTime);
        }
    }
}


struct ThreadDataSynapse
{
    vector<ExponentialSynapse>* synapse;
    float dt=0.1;
};

void updateSynapse(void* arg){
    ThreadDataSynapse* data = reinterpret_cast<ThreadDataSynapse*>(arg);
    vector<ExponentialSynapse>* synapse=data->synapse;
    float dt=data->dt;
    for (int i = 0; i < synapse->size(); i++)
    {
        (*synapse)[i].update(dt);
    }
}

int main() {
    srand(time(0));
    float dt=0.1;
    float currentTime=0.0;
    float stimuTime = 1000.0;
    auto start = high_resolution_clock::now();
    EInet net;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Initialization time: " << duration.count() << " ms" << endl;
    int flag1=0;
    int flag2=4;
    ThreadData dataE, dataI;
    ThreadDataSynapse dataE2E,dataE2I,dataI2I,dataI2E;
    ThreadPool *pool=new ThreadPool(10,10);

    dataE.neurons = &net.groupE;
    dataE.spikes = &net.spikesE;
    dataE.dt=dt;
    dataE.currentTime=currentTime;
    dataI.neurons = &net.groupI;
    dataI.spikes = &net.spikesI;
    dataI.dt=dt;
    dataI.currentTime=currentTime;
    dataE2E.synapse = &net.E2E;
    dataE2E.dt=dt;
    dataE2I.synapse = &net.E2I;
    dataE2I.dt=dt;
    dataI2I.synapse = &net.I2I;
    dataI2I.dt=dt;
    dataI2E.synapse = &net.I2E;
    dataI2E.dt=dt;
    auto start_stim = high_resolution_clock::now();
    int step = stimuTime / dt;
    for (int i = 0; i < step; i++)
    {
        currentTime+=dt;
        dataE.currentTime=currentTime;
        dataI.currentTime=currentTime;
        pool->pushJob(updateSynapse,&dataE2E,sizeof(dataE2E));
        pool->pushJob(updateSynapse,&dataE2I,sizeof(dataE2I));
        pool->pushJob(updateSynapse,&dataI2I,sizeof(dataI2I));
        pool->pushJob(updateSynapse,&dataI2E,sizeof(dataI2E));
    }
    auto end_stim = high_resolution_clock::now();
    auto duration_stim = duration_cast<milliseconds>(end_stim - start_stim);
    cout << "stimuTime: " << duration_stim.count() << " ms" << endl;
    net.saveSpikeTimes();

    return 0;
}