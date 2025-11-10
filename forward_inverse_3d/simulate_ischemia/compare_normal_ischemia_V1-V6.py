import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.visualize_tools import plot_val_on_mesh

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
step_per_timeframe = 4

leads = [' ', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

data_ischemia = np.load('forward_inverse_3d/data/simulate_ischemia/simulate_ischemia_{}.npz'.format(leads[1]))
data_normal = np.load('forward_inverse_3d/data/simulate_ischemia/simulate_normal_heart.npz')

v_data_ischemia = data_ischemia['v_data']
v_data_normal = data_normal['v_data']

standard12Lead_ischemia = data_ischemia['standard12Lead']
standard12Lead_normal = data_normal['standard12Lead']

import multiprocessing
p1 = multiprocessing.Process(target=plot_val_on_mesh, args=(mesh_file, v_data_ischemia[0],), kwargs={"target_cell": 2, "name": "v_ischemia", "title": "v on ventricle with ischemia", "f_val_flag": True})
p1.start()

ischemia_mask = np.where(np.abs(v_data_ischemia[0]+90) > 1, True, False)
ischemia_indices = np.where(ischemia_mask)[0]

# plot sequence of v
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
time = np.arange(0, v_data_ischemia.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
for i in range(9):
    ax = axs.flat[i]
    ax.plot(time, v_data_ischemia[:, ischemia_indices[i * len(ischemia_indices) // 9]], label='ischemia')
    ax.plot(time, v_data_normal[:, ischemia_indices[i * len(ischemia_indices) // 9]], label='normal', linestyle='--')
    ax.set_title('Ischemic Site {}'.format(i+1))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Transmembrane Potential (mV)')
    ax.legend()
    ax.grid(True)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.title('Transmembrane Potential in Ischemic Region Over Time')
plt.xlabel('Time Frame')
plt.ylabel('Transmembrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.show()

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_1 = np.arange(0, standard12Lead_ischemia.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
time_2 = np.arange(0, standard12Lead_normal.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
for i, ax in enumerate(axs.flat):
    ax.plot(time_1, standard12Lead_ischemia[:, i])
    ax.plot(time_2, standard12Lead_normal[:, i], linestyle='--')
    ax.legend(['ischemia', 'normal'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

p1.join()
