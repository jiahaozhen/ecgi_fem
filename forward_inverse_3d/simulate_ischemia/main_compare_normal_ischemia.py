# 基于有限元正过程比较正常心脏与缺血心脏的12导联心电图
import numpy as np
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_ischemia import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead
from utils.visualize_tools import plot_val_on_mesh, compare_standard_12_lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'

T = 500
step_per_timeframe = 8

center_ischemia = np.array([80.4, 19.7, -15.0])
radius_ischemia = 30
ischemia_epi_endo = [-1]
u_peak_ischemia_val = 0.9
u_rest_ischemia_val = 0.1

v_data_ischemia, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                              ischemia_flag=True, 
                                                              T=T,
                                                              center_ischemia=center_ischemia, 
                                                              radius_ischemia=radius_ischemia, 
                                                              ischemia_epi_endo=ischemia_epi_endo, 
                                                              u_peak_ischemia_val=u_peak_ischemia_val, 
                                                              u_rest_ischemia_val=u_rest_ischemia_val,
                                                              step_per_timeframe=step_per_timeframe)
v_data_normal, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                            ischemia_flag=False, 
                                                            T=T, 
                                                            step_per_timeframe=step_per_timeframe)

d_data_ischemia = compute_d_from_tmp(mesh_file, 
                                     v_data_ischemia, 
                                     ischemia_flag=True,
                                     center_ischemia=center_ischemia,
                                     ischemia_epi_endo=ischemia_epi_endo,
                                     radius_ischemia=radius_ischemia)
d_data_normal = compute_d_from_tmp(mesh_file, 
                                   v_data_normal, 
                                   ischemia_flag=False)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_ischemia = transfer_bsp_to_standard12lead(d_data_ischemia, leadIndex)
standard12Lead_normal = transfer_bsp_to_standard12lead(d_data_normal, leadIndex)

np.savez(r'forward_inverse_3d/data/simulate_ischemia/compare_normal_ischemia.npz',
         v_data_ischemia=v_data_ischemia,
         d_data_ischemia=d_data_ischemia,
         v_data_normal=v_data_normal,
         d_data_normal=d_data_normal,
         standard12Lead_ischemia=standard12Lead_ischemia,
         standard12Lead_normal=standard12Lead_normal)

import multiprocessing
p1 = multiprocessing.Process(target=plot_val_on_mesh, 
                             args=(mesh_file, v_data_ischemia[0]), 
                             kwargs={"target_cell": 2, 
                                     "name": "v_ischemia", 
                                     "title": "v on ventricle with ischemia", 
                                     "f_val_flag": True})
p2 = multiprocessing.Process(target=compare_standard_12_lead, 
                             args=(standard12Lead_normal, standard12Lead_ischemia), 
                             kwargs={'labels': ['Normal', 'Ischemia'],
                                     "step_per_timeframe": step_per_timeframe})
p1.start()
p2.start()

p1.join()
p2.join()
