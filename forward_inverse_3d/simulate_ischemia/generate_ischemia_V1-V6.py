import sys
import os

import numpy as np

sys.path.append('.')
from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 500
step_per_timeframe = 4
leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

ischemia_centers = np.array([[82.7, -7.8, -13.3], [88.3, 34.8, -3.4], 
                             [87.8, 80.4, -29.6], [74, 95.1, -30.7],
                             [37.5, 99.2, -27.4], [18.4, 80.6, -13.3]]).T

output_file_format = r'forward_inverse_3d/data/simulate_ischemia/simulate_ischemia_V{}.npz'
normal_output_file = r'forward_inverse_3d/data/simulate_ischemia/simulate_normal_heart.npz'

if os.path.exists(normal_output_file):
    print("Normal heart simulation data already exists. Skipping computation.")
else:
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe)
    d_data = compute_d_from_tmp(mesh_file, v_data)
    standard12Lead = transfer_bsp_to_standard12lead(d_data, leadIndex)
    np.savez(normal_output_file, v_data=v_data, d_data=d_data, standard12Lead=standard12Lead)

for i, center in enumerate(ischemia_centers.T):
    output_file = output_file_format.format(i+1)
    if os.path.exists(output_file):
        print(f"Results for ischemia V{i+1} already exist. Skipping computation.")
        continue
    
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T,
                                                                  center_ischemia=center,
                                                                  u_peak_ischemia_val=0.9,
                                                                  u_rest_ischemia_val=0.1,
                                                                  step_per_timeframe=step_per_timeframe)

    d_data = compute_d_from_tmp(mesh_file, v_data, center_ischemia=center, ischemia_flag=True)
    standard12Lead = transfer_bsp_to_standard12lead(d_data, leadIndex)

    
    np.savez(output_file, v_data=v_data, d_data=d_data,
             standard12Lead=standard12Lead)
    print(f"Results for ischemia V{i+1} saved to {output_file}")