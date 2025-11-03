# 基于有限元正过程比较正常心脏与缺血心脏的12导联心电图
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead
from utils.visualize_tools import plot_val_on_mesh

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 450
step_per_timeframe = 5

v_data_ischemia, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T, 
                                                              u_peak_ischemia_val=0.8, 
                                                              u_rest_ischemia_val=0.2, 
                                                              step_per_timeframe=step_per_timeframe)
v_data_normal, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=False, T=T, step_per_timeframe=step_per_timeframe)

d_data_ischemia = compute_d_from_tmp(mesh_file, v_data_ischemia, ischemia_flag=True)
d_data_normal = compute_d_from_tmp(mesh_file, v_data_normal)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_ischemia = transfer_bsp_to_standard12lead(d_data_ischemia, leadIndex)
standard12Lead_normal = transfer_bsp_to_standard12lead(d_data_normal, leadIndex)

import multiprocessing
p1 = multiprocessing.Process(target=plot_val_on_mesh, args=(mesh_file, v_data_ischemia[0],), kwargs={"target_cell": 2, "name": "v_ischemia", "title": "v on ventricle with ischemia", "f_val_flag": True})
p1.start()

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
