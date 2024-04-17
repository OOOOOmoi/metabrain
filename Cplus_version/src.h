#include <iostream>
#include <cmath>
#ifndef SRC_H
#define SRC_H
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
#endif // SRC_H