# 基于有限元正过程比较心脏躯干解耦耦合模型结果
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled import compute_d_from_tmp as compute_d_from_tmp_coupled
from forward_inverse_3d.forward.forward_decoupled import compute_d_from_tmp as compute_d_from_tmp_decoupled
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 500
step_per_timeframe = 2

v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=False, T=T, step_per_timeframe=step_per_timeframe)

d_data_coupled = compute_d_from_tmp_coupled(mesh_file, v_data, multi_flag=True)
d_data_decoupled = compute_d_from_tmp_decoupled(mesh_file, v_data, multi_flag=True)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_coupled = transfer_bsp_to_standard12lead(d_data_coupled, leadIndex)
standard12Lead_decoupled = transfer_bsp_to_standard12lead(d_data_decoupled, leadIndex)

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_1 = np.arange(0, standard12Lead_coupled.shape[0] / step_per_timeframe, 1/step_per_timeframe)
time_2 = np.arange(0, standard12Lead_decoupled.shape[0] / step_per_timeframe, 1/step_per_timeframe)
for i, ax in enumerate(axs.flat):
    ax.plot(time_1, standard12Lead_coupled[:, i])
    ax.plot(time_2, standard12Lead_decoupled[:, i], linestyle='--')
    ax.legend(['coupled', 'decoupled'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()