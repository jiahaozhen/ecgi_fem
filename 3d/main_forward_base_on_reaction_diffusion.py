import sys

import numpy as np
from main_forward_tmp import forward_tmp
from main_create_mesh_ecgsim_multi_conduct import create_mesh

sys.path.append('.')
# from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion
from reaction_diffusion.main_reaction_diffusion import compute_v_based_on_reaction_diffusion

gdim = 3
# if gdim == 2:
#     mesh_file = '2d/data/heart_torso.msh'
#     center_activation = np.array([4.0, 4.0])
#     radius_activation = 0.1
#     center_ischemia = np.array([4.0, 6.0])
#     radius_ischemia = 0.5
#     T = 40
# else:
#     mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
#     create_mesh(mesh_file, lc=40)
#     center_activation = np.array([57, 51.2, 15])
#     radius_activation = 5
#     center_ischemia = np.array([89.1, 40.9, -13.3])
#     radius_ischemia = 30
#     T = 40
# v_data, phi_1, phi_2 = compute_v_based_on_reaction_diffusion(
#     mesh_file=mesh_file, T=T, submesh_flag=True, ischemia_flag=True,
#     gdim=gdim, center_activation=center_activation, radius_activation=radius_activation,
#     center_ischemia=center_ischemia, radius_ischemia=radius_ischemia, 
#     data_argument=True, surface_flag=False
# )
mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
ischemia_flag = True
data_argument = True
v_data, phi_1, phi_2 = compute_v_based_on_reaction_diffusion(mesh_file, tau=10,
                                                             ischemia_flag=ischemia_flag, 
                                                             data_argument=data_argument)

# sample data
u_data = forward_tmp(mesh_file, v_data, gdim=gdim)

# save data
file_name_prefix = '2d/data/' if gdim == 2 else '3d/data/'
file_name_suffix = '_data_reaction_diffusion'

u_file_name = (file_name_prefix + 'u' + file_name_suffix
               + ('_ischemia' if ischemia_flag else '_normal')
               + ('_data_argument' if data_argument else '_no_data_argument') + '.npy')
v_file_name = (file_name_prefix + 'v' + file_name_suffix 
               + ('_ischemia' if ischemia_flag else '_normal')
               + ('_data_argument' if data_argument else '_no_data_argument') + '.npy')
phi_1_file_name = (file_name_prefix + 'phi_1' + file_name_suffix 
                   + ('_ischemia' if ischemia_flag else '_normal') + '.npy')
phi_2_file_name = (file_name_prefix + 'phi_2' + file_name_suffix 
                   + ('_ischemia' if ischemia_flag else '_normal') + '.npy')

np.save(u_file_name, u_data)
np.save(v_file_name, v_data)
np.save(phi_1_file_name, phi_1)
np.save(phi_2_file_name, phi_2)