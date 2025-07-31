import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.main_forward_tmp import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
v_data_epi_endo, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[-1, 0, 1], T=120)
v_data_epi, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[-1, 1], T=120)


d_data_epi_endo = compute_d_from_tmp(mesh_file, v_data=v_data_epi_endo)
d_data_epi = compute_d_from_tmp(mesh_file, v_data=v_data_epi)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_epi_endo = transfer_bsp_to_standard12lead(d_data_epi_endo, leadIndex)
standard12Lead_epi = transfer_bsp_to_standard12lead(d_data_epi, leadIndex)

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_normal = np.arange(0, standard12Lead_epi_endo.shape[0] / 10, 0.1)
time_ischemia = np.arange(0, standard12Lead_epi.shape[0] / 10, 0.1)
for i, ax in enumerate(axs.flat):
    ax.plot(time_normal, standard12Lead_epi_endo[:, i])
    ax.plot(time_ischemia, standard12Lead_epi[:, i], linestyle='--')
    ax.legend(['epi_endo', 'epi'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()