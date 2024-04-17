#include <iostream>
#include <random>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;


struct LifParams {
    float V_rest = -70.0;   // rest voltage
    float V_reset = -75.0;   // reset voltage
    float V_th = -50.0;  // threshold voltage
    float tau_m = 7.5;  // time constant
    float t_refractory = 2.0;  // refractory period
    float R = 0.05; // 电阻
} lifParamInit;

// LIF neuron model class
class LIFNeuron {
public:
    float tau_m;  // 膜时间常数
    float V_rest; // 静息电位
    float V_reset; // 重置电位
    float V_th;   // 阈值电位
    float R;      // 膜电阻
    float V;      // 膜电位
    float tau_ref; // 不应期时长
    float refractory_time; // 当前不应期剩余时间
    float input_current = 0.0;
    bool spiked;

    LIFNeuron()
    {
        tau_m = lifParamInit.tau_m;
        V_rest = lifParamInit.V_rest;
        V_reset = lifParamInit.V_reset;
        V_th = lifParamInit.V_th;
        R = lifParamInit.R;
        tau_ref = lifParamInit.t_refractory;
        V = -70.0;
        refractory_time = 0.0;
        spiked = false;
    }

    LIFNeuron(float _tau_m, 
              float _V_rest, 
              float _V_reset, 
              float _V_th, 
              float _R, 
              float _tau_ref,
              float _V)
    {
        tau_m = _tau_m;
        V_rest = _V_rest;
        V_reset = _V_reset;
        V_th = _V_th;
        R = _R;
        tau_ref = _tau_ref;
        V = _V;
        refractory_time = 0.0;
        spiked = false;
    }

    void receiveCurrent(float current) {
        input_current += current;
    }

    void update(float dt)
    {
        spiked = false;
        if (refractory_time > 0) 
        {
            refractory_time -= dt;
            V = V_reset;
        }
        else 
        {
            float total_current = input_current;
            float V_inf = V_rest + R * total_current;
            V += dt * (V_inf - V) / tau_m;

            if (V >= V_th) {
                // cout << "Spike!" << endl;
                spiked = true;
                V = V_reset;
                refractory_time = tau_ref;
            }
        }
        input_current = 0;
    }

    bool hasFired() {
        return spiked;
    }

    float getMembranePotential() {
        return V;
    }

};


// 指数型衰减突触模型(AMPA, GABA)
class ExponentialSynapse {
public:
    // ExponentialSynapse() {}
    LIFNeuron* pre;
    LIFNeuron* post;
    float g_max;
    float E_syn;
    float tau;
    float s; // 中间变量

    ExponentialSynapse(LIFNeuron* _pre, LIFNeuron* _post, float _g_max, float _E_syn, float _tau)
    {
        pre = _pre;
        post = _post;
        g_max = _g_max;
        E_syn = _E_syn;
        tau = _tau;
        s = 0.0;
    }

    void update(float dt) 
    {
        if (pre->hasFired()) {
            s += 1.0; // 突触前神经元fire时s++
        }
        // s += pre->hasFired();

        // 更新s和g的值
        s -= s / tau * dt; // 根据 tau 更新 s，电导的指数衰减
        
        // 计算突触电流
        float g_exp = g_max * s;
        float I_syn = g_exp * (E_syn - post->getMembranePotential());
        post->receiveCurrent(I_syn);
    }
};

class NMDASynapse {
public:
    LIFNeuron* pre;   // 前神经元
    LIFNeuron* post;  // 后神经元
    float g_max;     // 最大突触电导
    float tau_rise;  // 神经递质浓度上升时间常数
    float tau_decay; // 电导衰减时间常数
    float E_syn;     // 突触反转电位
    float s;         // 当前电导
    float x;         // 神经递质浓度
    float Mg;  // Mg2+ 浓度


    NMDASynapse(LIFNeuron* _pre, LIFNeuron* _post, float _g_max, float _tau_rise, float _tau_decay, float _E_syn, float _Mg)
    {
        pre = _pre;
        post = _post; 
        g_max = _g_max; 
        tau_rise = _tau_rise; 
        tau_decay = _tau_decay; 
        E_syn = _E_syn; 
        s = 0; 
        x = 0;
        Mg = _Mg;
    }

    void update(float dt) 
    {
        if (pre->hasFired()) {
            x += 1.0; // 突触前神经元发放动作电位时增加神经递质浓度
        }
        // x += pre->hasFired();

        // 更新 x 和 g 的值
        x -= dt * x / tau_rise; // 根据 τ_rise 更新 x
        s += dt * (-s / tau_decay + 0.5 * x * (1 - s)); // 根据 τ_decay 更新 g

        // 计算突触电流
        float g_NMDA = g_max * s;
        float V_post = post->getMembranePotential();
        float I_syn = g_NMDA * (E_syn - V_post) / (1 + Mg * exp(-0.062 * V_post) / 3.57);
        post->receiveCurrent(I_syn);
    }

private:

};


const int numNeuronsMax = 200;
const int numLipExcA = 200;
const int numLipExcB = 200;
const int numLipInh = 100;

class EInet
{
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

    EInet()
    {
        createneuron();
        createsynapse();
    }


    void createneuron()
    {
        for (int i = 0; i < numE; i++)
        {
            groupE.emplace_back(LIFNeuron(lifpara.tau_m,lifpara.V_rest,
                                lifpara.V_reset,lifpara.V_th,lifpara.R,lifpara.t_refractory,-70.0));
        }
        for (int i = 0; i < numI; i++)
        {
            groupI.emplace_back(LIFNeuron(lifpara.tau_m,lifpara.V_rest,
                                lifpara.V_reset,lifpara.V_th,lifpara.R,lifpara.t_refractory,-70.0));
        }
        
    }
    void createsynapse()
    {
        for (auto& preneuron : groupE)
        {
            for (auto& postneuron : groupE)
            {
                E2E.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_AMPA,E_AMPA,tau_AMPA));
            }
        }
        for (auto& preneuron : groupE)
        {
            for (auto& postneuron : groupI)
            {
                E2I.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_AMPA,E_AMPA,tau_AMPA));
            }
        }
        for (auto& preneuron : groupI)
        {
            for (auto& postneuron : groupE)
            {
                I2E.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_GABA,E_GABA,tau_GABA));
            }
        }
        for (auto& preneuron : groupI)
        {
            for (auto& postneuron : groupI)
            {
                I2I.emplace_back(ExponentialSynapse(&preneuron,&postneuron,g_GABA,E_GABA,tau_GABA));
            }
        }
    }
    void update()
    {
        for(auto syn : E2E){
            syn.update(dt);
        }
        for(auto syn : E2I){
            syn.update(dt);
        }
        for(auto syn : I2I){
            syn.update(dt);
        }
        for(auto syn : I2E){
            syn.update(dt);
        }
        for(auto neu : groupE){
            neu.update(dt)
        }
        for(auto neu : groupI){
            neu.update(dt)
        }
    }
};


class ColorDecision
{
    public:
    
    // float scale = 1.0;

    float V_init = -70.0;

    float poissonNoiseFreq = 1000.0;

    float g_noise_E = 2;
    float g_noise_I = 3;

    float g_EE_AMPA = 0.01;
    float g_EI_AMPA = 0.01;

    float g_IE = 0.8;
    float g_II = 0.3;

    float g_EE_NMDA = 0.5;
    float g_EI_NMDA = 0.3;

    float E_syn_AMPA = 0.0;
    float E_syn_NMDA = 0.0;
    float Mg = 1.0;
    float E_syn_GABA = -70.0;

    float tau_AMPA = 2.0;
    float tau_GABA = 5.0;
    float tau_rise_NMDA = 2.0;
    float tau_decay_NMDA = 100;

    float dt = 1.0;
    
    // 定义神经元群
    vector<LIFNeuron> lipPopExcA;   // LIP脑区兴奋性神经元群A
    vector<LIFNeuron> lipPopExcB;   // LIP脑区兴奋性神经元群B
    vector<LIFNeuron> lipPopInh;    // LIP脑区抑制性神经元群

    // 定义突触, 其中 '==>': AMPA, '-=>': NDMA, '-->': GABA
    vector<ExponentialSynapse> lipExcA2ExcASyn_AMPA;  // A ==> A
    vector<ExponentialSynapse> lipExcA2InhSyn_AMPA;   // A ==> I

    vector<NMDASynapse> lipExcA2ExcASyn_NMDA;  // A -=> A
    vector<NMDASynapse> lipExcA2InhSyn_NMDA;   // A -=> I

    vector<ExponentialSynapse> lipExcB2ExcBSyn_AMPA;  // B ==> B
    vector<ExponentialSynapse> lipExcB2InhSyn_AMPA;   // B ==> I

    vector<NMDASynapse> lipExcB2ExcBSyn_NMDA;  // B -=> B
    vector<NMDASynapse> lipExcB2InhSyn_NMDA;   // B -=> I

    vector<ExponentialSynapse> lipInh2ExcASyn_GABA;   // I --> A
    vector<ExponentialSynapse> lipInh2ExcBSyn_GABA;   // I --> B
    vector<ExponentialSynapse> lipInh2InhSyn_GABA;    // I --> I

    ColorDecision()
    {
        initializeNeurons();
        createSynapses();

        // cout << lipPopExcA.size() << endl;
        // cout << lipPopExcB.size() << endl;
        // cout << lipPopInh.size() << endl;

        // cout << lipPopExcA[0].getMembranePotential() << endl;
        // cout << lipPopExcB[0].tau_ref << endl;

        // cout << lipExcA2InhSyn_AMPA.size() << endl;
        // cout << lipExcA2InhSyn_AMPA[6].pre->getMembranePotential() << endl;
        // cout << lipExcB2ExcBSyn_NMDA[5].Mg << endl;
        // cout << lipInh2ExcASyn_GABA[0].E_syn << endl;
        // cout << "Init complete!" << endl;
    }

    void initializeNeurons()
    {
        // 初始化神经元
        for (int i=0; i<numLipExcA; i++)
        {
            lipPopExcA.emplace_back(lifParamInit.tau_m, lifParamInit.V_rest, lifParamInit.V_reset, lifParamInit.V_th, lifParamInit.R, lifParamInit.t_refractory, V_init);
            lipPopExcB.emplace_back(lifParamInit.tau_m, lifParamInit.V_rest, lifParamInit.V_reset, lifParamInit.V_th, lifParamInit.R, lifParamInit.t_refractory, V_init);
            if (i < 100) {
                lipPopInh.emplace_back(lifParamInit.tau_m, lifParamInit.V_rest, lifParamInit.V_reset, lifParamInit.V_th, lifParamInit.R, lifParamInit.t_refractory, V_init);
            }
        }
    }

    void createSynapses() 
    {
        // lipPopExcA自连接, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcA) {
            for (auto& post_neuron : lipPopExcA) {
                lipExcA2ExcASyn_AMPA.emplace_back(&pre_neuron, &post_neuron, g_EE_AMPA, E_syn_AMPA, tau_AMPA); // AMPASynapse 参数
                lipExcA2ExcASyn_NMDA.emplace_back(&pre_neuron, &post_neuron, g_EE_NMDA, tau_rise_NMDA, tau_decay_NMDA, E_syn_NMDA, Mg); // NMDASynapse 参数
            }
        }

        // lipPopExcA -> lipPopInh, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcA) {
            for (auto& post_neuron : lipPopInh) {
                lipExcA2InhSyn_AMPA.emplace_back(&pre_neuron, &post_neuron, g_EI_AMPA, E_syn_AMPA, tau_AMPA); // AMPASynapse 参数
                lipExcA2InhSyn_NMDA.emplace_back(&pre_neuron, &post_neuron, g_EI_NMDA, tau_rise_NMDA, tau_decay_NMDA, E_syn_NMDA, Mg); // NMDASynapse 参数
            }
        }

        // lipPopExcB自连接, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcB) {
            for (auto& post_neuron : lipPopExcB) {
                lipExcB2ExcBSyn_AMPA.emplace_back(&pre_neuron, &post_neuron, g_EE_AMPA, E_syn_AMPA, tau_AMPA); // AMPASynapse 参数
                lipExcB2ExcBSyn_NMDA.emplace_back(&pre_neuron, &post_neuron, g_EE_NMDA, tau_rise_NMDA, tau_decay_NMDA, E_syn_NMDA, Mg); // NMDASynapse 参数
            }
        }

        // lipPopExcB -> lipPopInh, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcB) {
            for (auto& post_neuron : lipPopInh) {
                lipExcB2InhSyn_AMPA.emplace_back(&pre_neuron, &post_neuron, g_EI_AMPA, E_syn_AMPA, tau_AMPA); // AMPASynapse 参数
                lipExcA2ExcASyn_NMDA.emplace_back(&pre_neuron, &post_neuron, g_EI_NMDA, tau_rise_NMDA, tau_decay_NMDA, E_syn_NMDA, Mg); // NMDASynapse 参数
            }
        }

        // lipPopInh -> lipPopExcA, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopExcA) {
                lipInh2ExcASyn_GABA.emplace_back(&pre_neuron, &post_neuron, g_IE, E_syn_GABA, tau_GABA); // GABASynapse 参数
            }
        }

        // lipPopInh -> lipPopExcB, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopExcB) {
                lipInh2ExcBSyn_GABA.emplace_back(&pre_neuron, &post_neuron, g_IE, E_syn_GABA, tau_GABA); // GABASynapse 参数
            }
        }

        // lipPopInh自连接, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopInh) {
                lipInh2InhSyn_GABA.emplace_back(&pre_neuron, &post_neuron, g_II, E_syn_GABA, tau_GABA); // GABASynapse 参数
            }
        }

    }

    void update(int perStep)
    {
        for (int i=0; i<perStep; i++) //f=jax.numpy.array(t*fps/1000,int)
        {
            // 突触更新
            // lipPopExcA自连接突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcA2ExcASyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcA2ExcASyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcA -> lipPopInh突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcA2InhSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcA2InhSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcB自连接突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcB2ExcBSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcB2ExcBSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcB -> lipPopInh突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcB2InhSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcB2InhSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopInh -> lipPopExcA突触更新, GABA
            for (auto& gabaSyn : lipInh2ExcASyn_GABA) {
                gabaSyn.update(dt);
            }

            // lipPopInh -> lipPopExcB突触更新, GABA
            for (auto& gabaSyn : lipInh2ExcBSyn_GABA) {
                gabaSyn.update(dt);
            }

            // lipPopInh自连接突触更新, GABA
            for (auto& gabaSyn : lipInh2InhSyn_GABA) {
                gabaSyn.update(dt);
            }            

            // 神经元更新
            for (auto& neuronA : lipPopExcA) {
                neuronA.update(dt);
            }
            for (auto& neuronB : lipPopExcB) {
                neuronB.update(dt);
            }
            for (auto& neuronI : lipPopInh) {
                neuronI.update(dt);
            }
        }
    }
};
int main()
{
    
    cout << "ColorDecision model init!" << endl;
    auto start_init = high_resolution_clock::now();
    ColorDecision cd;
    auto end_init = high_resolution_clock::now();
    auto dur_init = duration_cast<microseconds>(end_init - start_init);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model init: " << dur_init.count() / 1000.0 << " ms" << endl;          

    int stimFrame = 1000/9;
    int perStep = 9;
    auto start_update = high_resolution_clock::now();
    for (int i=0; i<stimFrame; i++) {
        cd.update(perStep);
    }
    
    auto end_update = high_resolution_clock::now();
    auto dur_update = duration_cast<microseconds>(end_update - start_update);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model update: " << dur_update.count() / 1000.0 << " ms" << endl;
}


