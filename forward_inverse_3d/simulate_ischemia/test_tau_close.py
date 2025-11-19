'''
测试 tau_close 在空间中是否变化对 12 导联心电图的影响
结论 tau_close 的变化对 12 导联心电图影响较大 结果会偏离正常范围
'''
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_ischemia import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 450
step_per_timeframe = 2

v_data_no_vary, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe, tau_close_vary=False)
v_data_vary, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe, tau_close_vary=True)

d_data_no_vary = compute_d_from_tmp(mesh_file, v_data_no_vary)
d_data_vary = compute_d_from_tmp(mesh_file, v_data_vary)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_no_vary = transfer_bsp_to_standard12lead(d_data_no_vary, leadIndex)
standard12Lead_vary = transfer_bsp_to_standard12lead(d_data_vary, leadIndex)


fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_1 = np.arange(0, standard12Lead_no_vary.shape[0] / step_per_timeframe, 1/step_per_timeframe)
time_2 = np.arange(0, standard12Lead_vary.shape[0] / step_per_timeframe, 1/step_per_timeframe)
for i, ax in enumerate(axs.flat):
    ax.plot(time_1, standard12Lead_no_vary[:, i])
    ax.plot(time_2, standard12Lead_vary[:, i], linestyle='--')
    ax.legend(['no vary in space', 'vary in space'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
