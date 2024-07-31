import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_spike_data(file_name):
    spike_data = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip():
                spikes = [float(s) for s in line.split()]
                spike_data.append(spikes)
    return spike_data

def plot_spike_times(spike_data, title, ylim):
    for i, spikes in enumerate(spike_data):
        plt.plot(spikes, [i+1] * len(spikes), '.',markersize='0.5', color='black')
    plt.title(title)
    plt.ylim(ylim)
    plt.ylabel('neuron index')


def plot_average_spike_rates(ax, spike_data, window_length_ms, title):
    max_time = max(max(spikes) if spikes else 0 for spikes in spike_data)
    time_bins = np.arange(0, max_time + window_length_ms, window_length_ms)
    all_rates = []

    for spikes in spike_data:
        spike_counts, _ = np.histogram(spikes, bins=time_bins)
        spike_rates = spike_counts / (window_length_ms / 1000.0)  # Hz
        all_rates.append(spike_rates)

    if all_rates:
        average_rates = np.mean(all_rates, axis=0)
        ax.plot(time_bins[:-1], average_rates, color='red', label='Average Rate')
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Average Spike Rate (Hz)')
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)


parser = argparse.ArgumentParser(description='Plot spike times and average spike rates.')
parser.add_argument('--exc_file', type=str, required=True, help='Path to the excitatory neuron spike times file')
parser.add_argument('--inh_file', type=str, required=True, help='Path to the inhibitory neuron spike times file')
parser.add_argument('--output_file', type=str, required=True, help='Path to save the output plot')
# parser.add_argument('--window_length_ms', type=float, default=10.0, help='Window length in ms for calculating spike rates')

args = parser.parse_args()

# 设置窗口大小
window_length_ms = 5

# 读取数据
group_1_data = read_spike_data(args.exc_file)
group_2_data = read_spike_data(args.inh_file)

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex='col', sharey='row')

# 绘制散点图
plt.sca(axs[0,0])
plot_spike_times(group_1_data, 'Spike Times for GroupE',4000)
plt.sca(axs[0,1])
plot_spike_times(group_2_data, 'Spike Times for GroupI',1000)

# 绘制频率图
plot_average_spike_rates(axs[1, 0], group_1_data, window_length_ms, 'Spike Rates for GroupE')
plot_average_spike_rates(axs[1, 1], group_2_data, window_length_ms, 'Spike Rates for GroupI')

for ax_row in axs:
    for ax in ax_row:
        ax.autoscale(axis='y')

plt.tight_layout()
plt.savefig(args.output_file)
