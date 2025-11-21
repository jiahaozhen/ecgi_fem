
import numpy as np
import matplotlib.pyplot as plt

def plot_surface_d():
    file_path = 'machine_learning/data/dataset/d_surface_dataset/ischemia_d_surface_part_000.npz'
    with np.load(file_path) as data:
        X = data['X']
        y = data['y']
    print('surface_d X shape:', X.shape)
    print('surface_d y shape:', y.shape)
    sample_idx = 0
    time_idx = 0
    sample = X[sample_idx, time_idx]
    plt.figure(figsize=(8, 4))
    plt.imshow(sample, aspect='auto', cmap='jet')
    plt.colorbar(label='Voltage')
    plt.title(f'surface_d Sample {sample_idx}, Time {time_idx}, Label: {y[sample_idx]}')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.tight_layout()
    plt.show()

def plot_d_standard():
    file_path = 'machine_learning/data/dataset/d_standard_dataset/ischemia_d_standard_part_000.npz'
    with np.load(file_path) as data:
        X = data['X']
        y = data['y']
    
    sample_idx = 0
    # 画一个导联随时间变化
    lead_idx = 10  # 可调整
    lead_curve = X[sample_idx, :, lead_idx]
    plt.figure()
    plt.plot(lead_curve)
    plt.title(f'd_standard {lead_idx}')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.tight_layout()
    plt.show()

def plot_v():
    file_path = 'machine_learning/data/dataset/v_dataset/ischemia_v_part_000.npz'
    with np.load(file_path) as data:
        X = data['X']
        y = data['y']
    sample_idx = 10
    # 画一个节点随时间变化
    node_idx = 10  # 可调整
    node_curve = X[sample_idx, :, node_idx]
    plt.figure()
    plt.plot(node_curve)
    plt.title(f'v {node_idx}')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plot_surface_d()
    # plot_d_standard()
    plot_v()
