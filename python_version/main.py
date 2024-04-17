import numpy as np
import time
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self,param_list):
        #参数初始化
        self.V_th=param_list["V_th"]
        self.V_reverse=param_list["V_reverse"]
        self.V_rest=param_list["V_rest"]
        self.refractory=param_list["refractory"]
        self.tau_m=param_list["tau_m"]
        self.Cm=param_list["Cm"]
        self.R=self.tau_m/self.Cm
        if param_list["method"]=="normal":
            self.V=self.V_rest+np.random.normal(0,param_list["sigma"],1)
        else:
            self.V=self.V_rest
        self.input_current=0
        self.spike=False
        self.ref_reftime=0
        self.spikes=[]

    def recive(self,current):
        self.input_current+=current

    def fired(self):
        return self.spike

    def get_voltage(self):
        return self.V
    
    def update(self):
        if self.ref_reftime>0:
            self.ref_reftime-=dt
            self.V=self.V_rest
            self.spike=False
        else:
            total_current=self.input_current
            V_inf=self.V_reverse+self.R*total_current
            self.V+=dt*(V_inf-self.V)/self.tau_m#欧拉法
            if self.V>=self.V_th:
                self.spike=True
                self.V=self.V_rest
                self.ref_reftime=self.refractory
        if self.spike:
            self.spikes.append(current_time)
        self.input_current=0


class ExponentialSynapse:
    def __init__(self,pre,post,paramlist):
        self.pre=pre
        self.post=post
        self.E_syn=paramlist["E_syn"]
        self.tau=paramlist["tau"]
        self.g_max=paramlist["g_max"]
        self.s=0
    
    def update(self):
        if self.pre.fired():
            self.s+=1
        self.s-=self.s/self.tau*dt
        I_syn = self.s*self.g_max * (self.E_syn - self.post.get_voltage())
        self.post.recive(I_syn)
        



class NMDASynapse:
    def __init__(self,pre,post,paramlist):
        self.pre=pre
        self.post=post
        self.g_max=paramlist["g_max"]
        self.tau_rise=paramlist["tau_rise"]
        self.tau_decay=paramlist["tau_decay"]
        self.E_syn=paramlist["E_syn"]
        self.Mg=paramlist["Mg"]
        self.s=0
        self.x=0
    def update(self):
        if self.pre.fired():
            self.x+=1
        self.x-=dt*self.x/self.tau_rise
        self.s+=dt*(-self.s / self.tau_decay + 0.5 * self.x * (1 - self.s))
        g_NMDA=self.g_max*self.s
        V_post=self.post.get_voltage()
        I_syn = g_NMDA * (V_post - self.E_syn) / (1 + self.Mg * np.exp(-0.062 * V_post) / 3.57);
        self.post.recive(I_syn)

class EInet:
    def __init__(self):
        self.initializeNeurons()
        self.createSynapses()

    LIFparam=dict(V_th=-50,V_rest=-60,V_reverse=-60,
              tau_m=20,refractory=5,Cm=20,method="normal",sigma=4,input=12)
    AMPAparam=dict(E_syn=0,tau=5,g_max=0.3)
    GABAparam=dict(E_syn=-80,tau=10,g_max=3.2)
    groupE=[]
    groupI=[]
    E2E=[]
    E2I=[]
    I2E=[]
    I2I=[]
    fixprob=0.02

    def initializeNeurons(self):
        for i in range(numE):
            neuron=LIFNeuron(self.LIFparam)
            self.groupE.append(neuron)

        for i in range(numI):
            neuron=LIFNeuron(self.LIFparam)
            self.groupI.append(neuron)

    def createSynapses(self):
        #E2E
        for preneuron in self.groupE:
            for postneuron in self.groupE:
                if np.random.uniform()<self.fixprob:
                    syn=ExponentialSynapse(preneuron,postneuron,self.AMPAparam)
                    self.E2E.append(syn)
        #E2I
        for preneuron in self.groupE:
            for postneuron in self.groupI:
                if np.random.uniform()<self.fixprob:
                    syn=ExponentialSynapse(preneuron,postneuron,self.AMPAparam)
                    self.E2I.append(syn)
        #I2I
        for preneuron in self.groupI:
            for postneuron in self.groupI:
                if np.random.uniform()<self.fixprob:
                    syn=ExponentialSynapse(preneuron,postneuron,self.GABAparam)
                    self.I2I.append(syn)
        #I2E
        for preneuron in self.groupI:
            for postneuron in self.groupE:
                if np.random.uniform()<self.fixprob:
                    syn=ExponentialSynapse(preneuron,postneuron,self.GABAparam)
                    self.I2E.append(syn)

    def update(self):
        for syn in self.E2E:
            syn.update()
        for syn in self.E2I:
            syn.update()
        for syn in self.I2I:
            syn.update()
        for syn in self.I2E:
            syn.update()
        
        for neuron in self.groupE:
            neuron.recive(12)
            neuron.update()
        for neuron in self.groupI:
            neuron.recive(12)
            neuron.update()



numE=4000
numI=1000
dt=0.1
sim_time=100
print("start init")
s=time.time()
net=EInet()
e=time.time()
print(f"initialize time {e-s}s")

print("start simulation")
s=time.time()
current_time=0
for i in range(int(sim_time/dt)):
    net.update()
    current_time+=dt

e=time.time()
print(f"simulation time {e-s}s")

#save spike
# all_spikes = [neuron.spikes for neuron in net.groupE+net.groupI]
# spike_times_array = np.array(all_spikes, dtype=object)
# np.save('spike_times.npy', spike_times_array)


#draw spike
print("start draw fig")
fig, ax = plt.subplots()
for idx, neuron in enumerate(net.groupI):
    for spike_time in neuron.spikes:
        ax.plot(spike_time, idx, 'k.', markersize=2)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
fig.savefig('neuron_spikes_groupI.png')


fig, ax = plt.subplots()
for idx, neuron in enumerate(net.groupE):
    for spike_time in neuron.spikes:
        ax.plot(spike_time, idx, 'k.', markersize=2)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
fig.savefig('neuron_spikes_groupE.png')  