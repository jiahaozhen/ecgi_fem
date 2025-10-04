# 观察心肌缺血不同区域（外膜，中间，内膜）对12导联心电图的影响
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.main_forward_tmp import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
# v_data_normal, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=False, T=120)
# v_data_epi, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[1], T=300)
# v_data_endo, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[-1], T=300)
# v_data_mid, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[0], T=300)
v_data_epi_mid, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[0, 1], T=300)
v_data_endo_mid, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[-1, 0], T=300)
# v_data_epi_mid_endo, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, ischemia_epi_endo=[-1, 0, 1], T=300)

# d_data_normal = compute_d_from_tmp(mesh_file, v_data_normal)
# d_data_epi = compute_d_from_tmp(mesh_file, v_data_epi)
# d_data_endo = compute_d_from_tmp(mesh_file, v_data_endo)
# d_data_mid = compute_d_from_tmp(mesh_file, v_data_mid)
d_data_epi_mid = compute_d_from_tmp(mesh_file, v_data_epi_mid)
d_data_endo_mid = compute_d_from_tmp(mesh_file, v_data_endo_mid)
# d_data_epi_mid_endo = compute_d_from_tmp(mesh_file, v_data_epi_mid_endo)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
# standard12Lead_normal = transfer_bsp_to_standard12lead(d_data_normal, leadIndex)
# standard12Lead_epi = transfer_bsp_to_standard12lead(d_data_epi, leadIndex)
# standard12Lead_endo = transfer_bsp_to_standard12lead(d_data_endo, leadIndex)
# standard12Lead_mid = transfer_bsp_to_standard12lead(d_data_mid, leadIndex)
standard12Lead_epi_mid = transfer_bsp_to_standard12lead(d_data_epi_mid, leadIndex)
standard12Lead_endo_mid = transfer_bsp_to_standard12lead(d_data_endo_mid, leadIndex)
# standard12Lead_epi_mid_endo = transfer_bsp_to_standard12lead(d_data_epi_mid_endo, leadIndex)

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_1 = np.arange(0, standard12Lead_endo_mid.shape[0] / 10, 0.1)
time_2 = np.arange(0, standard12Lead_endo_mid.shape[0] / 10, 0.1)
for i, ax in enumerate(axs.flat):
    ax.plot(time_1, standard12Lead_endo_mid[:, i])
    ax.plot(time_2, standard12Lead_epi_mid[:, i], linestyle='--')
    ax.legend(['endo', 'epi'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()