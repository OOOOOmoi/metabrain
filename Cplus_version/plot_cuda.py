import numpy as np
import matplotlib.pyplot as plt

def read_spike_data(file_path):
    spike_data_exc = []
    spike_data_inh = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Excitatory Neuron"):
                spike_times = next(file).strip().split()
                spike_times = list(map(float, spike_times))
                spike_data_exc.append(spike_times)
            elif line.startswith("Inhibitory Neuron"):
                spike_times = next(file).strip().split()
                spike_times = list(map(float, spike_times))
                spike_data_inh.append(spike_times)
    return spike_data_exc, spike_data_inh

def compute_firing_rate(spike_data, window_size=10, duration=100):
    bins = np.arange(0, duration + window_size, window_size)
    firing_rates = []
    for neuron_spikes in spike_data:
        counts, _ = np.histogram(neuron_spikes, bins)
        firing_rates.append(counts / (window_size / 1000.0))  # 转换为每秒的放电频率
    mean_firing_rate = np.mean(firing_rates, axis=0)
    return bins[:-1], mean_firing_rate

def plot_spike_times(spike_data_exc, spike_data_inh):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制Excitatory Neurons的spike times
    for i, spikes in enumerate(spike_data_exc):
        axs[0, 0].plot(spikes, [i+1] * len(spikes), '.', color='blue')
    axs[0, 0].set_title('Excitatory Neurons Spike Times')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Neuron Index')
    axs[0, 0].grid(True)
    
    # 绘制Inhibitory Neurons的spike times
    for i, spikes in enumerate(spike_data_inh):
        axs[0, 1].plot(spikes, [i+1] * len(spikes), '.', color='red')
    axs[0, 1].set_title('Inhibitory Neurons Spike Times')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylabel('Neuron Index')
    axs[0, 1].grid(True)

    # 计算和绘制Excitatory Neurons的平均放电频率
    bins, mean_firing_rate_exc = compute_firing_rate(spike_data_exc)
    axs[1, 0].plot(bins, mean_firing_rate_exc, color='blue')
    axs[1, 0].set_title('Average Firing Rate of Excitatory Neurons')
    axs[1, 0].set_xlabel('Time (ms)')
    axs[1, 0].set_ylabel('Firing Rate (Hz)')
    axs[1, 0].grid(True)

    # 计算和绘制Inhibitory Neurons的平均放电频率
    bins, mean_firing_rate_inh = compute_firing_rate(spike_data_inh)
    axs[1, 1].plot(bins, mean_firing_rate_inh, color='red')
    axs[1, 1].set_title('Average Firing Rate of Inhibitory Neurons')
    axs[1, 1].set_xlabel('Time (ms)')
    axs[1, 1].set_ylabel('Firing Rate (Hz)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('EI_net_cuda.png')

# 文件路径
file_path = '/home/yangjinhao/stucture/Cplus_version/spike_times.txt'

# 读取数据
spike_data_exc, spike_data_inh = read_spike_data(file_path)

# 绘制图像
plot_spike_times(spike_data_exc, spike_data_inh)
