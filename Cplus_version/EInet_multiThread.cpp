#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "src.h"
#include <time.h>
#include <fstream>

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
    int numE=4000;
    int numI=1000;
    
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
    void update(float currentTime){
        for(auto &syn : E2E){
            syn.update(dt);
        }
        for(auto &syn : E2I){
            syn.update(dt);
        }
        for(auto &syn : I2I){
            syn.update(dt);
        }
        for(auto &syn : I2E){
            syn.update(dt);
        }
        for(size_t i = 0; i < groupE.size(); ++i){
            groupE[i].receiveCurrent(input);
            groupE[i].update(dt);
            if (groupE[i].hasFired()) {
                spikesE[i].push_back(currentTime);
            }
        }
        for(size_t i = 0; i < groupI.size(); ++i){
            groupI[i].receiveCurrent(input);
            groupI[i].update(dt);
            if (groupI[i].hasFired()) {
                spikesI[i].push_back(currentTime);
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
};

void* updateNeurons(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    vector<LIFNeuron>* neurons = data->neurons;

    for (size_t j = 0; j < neurons->size(); ++j) {
        (*neurons)[j].receiveCurrent(12.0);
        (*neurons)[j].update(dt);
        if ((*neurons)[j].hasFired()) {
            (*spikes)[j].push_back(currentTime);
        }
    }

    pthread_exit(NULL);
}


struct ThreadDataSynapse
{
    vector<ExponentialSynapse>* synapse;
};

void* updateSynapse(void* arg){

}

int main() {
    srand(time(0));
    float stimuTime = 500.0;
    auto start = high_resolution_clock::now();
    EInet net;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Initialization time: " << duration.count() << " ms" << endl;

    pthread_t threadE, threadI;
    pthread_t syn1,syn2,syn3,syn4;
    ThreadData dataE, dataI;
    ThreadDataSynapse dataE2E,dataE2I,dataI2I,dataI2E;

    dataE.neurons = &net.groupE;
    dataI.neurons = &net.groupI;
    dateE2E.syn

    for (int i = 0; i < step; i++)
    {
        pthread_create(&syn1, NULL, updateSynapse, &synapse1);
        pthread_create(&syn2, NULL, updateSynapse, &synapse2);
        pthread_create(&syn3, NULL, updateSynapse, &synapse3);
        pthread_create(&syn4, NULL, updateSynapse, &synapse4);
        pthread_join();
        pthread_create(&threadE, NULL, updateNeurons, &dataE);
    }
    
    if (pthread_create(&threadE, NULL, updateNeurons, &dataE)) {
        cerr << "Error: Unable to create thread for groupE" << endl;
        return -1;
    }

    if (pthread_create(&threadI, NULL, updateNeurons, &dataI)) {
        cerr << "Error: Unable to create thread for groupI" << endl;
        return -1;
    }

    pthread_join(threadE, NULL);
    pthread_join(threadI, NULL);

    net.saveSpikeTimes();

    return 0;
}