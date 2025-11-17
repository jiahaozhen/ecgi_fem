import numpy as np
import matplotlib.pyplot as plt
from utils.visualize_tools import plot_val_on_mesh, compare_standard_12_lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
step_per_timeframe = 8

leads = [' ', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# data_ischemia = np.load('forward_inverse_3d/data/simulate_ischemia/simulate_ischemia_{}.npz'.format(leads[1]))
# data_normal = np.load('forward_inverse_3d/data/simulate_ischemia/simulate_normal_heart.npz')

# v_data_ischemia = data_ischemia['v_data']
# v_data_normal = data_normal['v_data']

# standard12Lead_ischemia = data_ischemia['standard12Lead']
# standard12Lead_normal = data_normal['standard12Lead']

data_compare = np.load('forward_inverse_3d/data/simulate_ischemia/compare_normal_ischemia.npz')
v_data_ischemia = data_compare['v_data_ischemia']
d_data_ischemia = data_compare['d_data_ischemia']
v_data_normal = data_compare['v_data_normal']
d_data_normal = data_compare['d_data_normal']
standard12Lead_ischemia = data_compare['standard12Lead_ischemia']
standard12Lead_normal = data_compare['standard12Lead_normal']

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

compare_standard_12_lead(standard12Lead_normal, standard12Lead_ischemia,
                         labels=['Normal', 'Ischemia'], 
                         step_per_timeframe=step_per_timeframe)

p1.join()
