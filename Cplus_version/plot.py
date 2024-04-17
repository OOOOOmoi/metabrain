import matplotlib.pyplot as plt

# 加载spike时间数据
with open('/home/yangjinhao/stucture/Cplus_version/E_spike_times.txt', 'r') as file:
    spike_data = []
    for line in file:
        spikes = [float(s) for s in line.split()]
        spike_data.append(spikes)

# 绘制spike时间
plt.figure(figsize=(10, 5))
for i, spikes in enumerate(spike_data):
    plt.plot(spikes, [i+1] * len(spikes), '.', color='blue')
plt.title('Spike Times of Neurons')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.grid(True)
plt.savefig('/home/yangjinhao/stucture/Cplus_version/E_spike_times.png')


# 加载spike时间数据
with open('/home/yangjinhao/stucture/Cplus_version/I_spike_times.txt', 'r') as file:
    spike_data = []
    for line in file:
        spikes = [float(s) for s in line.split()]
        spike_data.append(spikes)

# 绘制spike时间
plt.figure(figsize=(10, 5))
for i, spikes in enumerate(spike_data):
    plt.plot(spikes, [i+1] * len(spikes), '.', color='blue')
plt.title('Spike Times of Neurons')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.grid(True)
plt.savefig('/home/yangjinhao/stucture/Cplus_version/I_spike_times.png')