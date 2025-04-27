import sys

import numpy as np
import matplotlib.pyplot as plt
from main_forward_tmp import compute_d_from_tmp

sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
v_data_normal, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=False)
v_data_ischemia, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True)

d_data_normal = compute_d_from_tmp(mesh_file, v_data=v_data_normal)
d_data_ischemia = compute_d_from_tmp(mesh_file, v_data=v_data_ischemia)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_normal = transfer_bsp_to_standard12lead(d_data_normal, leadIndex)
standard12Lead_ischemia = transfer_bsp_to_standard12lead(d_data_ischemia, leadIndex)

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_normal = np.arange(0, standard12Lead_normal.shape[0] / 10, 0.1)
time_ischemia = np.arange(0, standard12Lead_ischemia.shape[0] / 10, 0.1)
for i, ax in enumerate(axs.flat):
    ax.plot(time_normal, standard12Lead_normal[:, i])
    ax.plot(time_ischemia, standard12Lead_ischemia[:, i], linestyle='--')
    ax.legend(['normal', 'ischemia'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
