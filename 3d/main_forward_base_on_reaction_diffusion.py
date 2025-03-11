import sys

import numpy as np
from main_forward_tmp import forward_tmp
from main_create_mesh_ecgsim_multi_conduct import create_mesh

sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion

gdim = 3
if gdim == 2:
    mesh_file = '2d/data/heart_torso.msh'
    center_activation = np.array([4.0, 4.0])
    radius_activation = 0.1
    center_ischemic = np.array([4.0, 6.0])
    radius_ischemic = 0.5
    T = 40
else:
    mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    create_mesh(mesh_file, lc = 40)
    center_activation = np.array([57, 51.2, 15])
    radius_activation = 5
    center_ischemic = np.array([89.1, 40.9, -13.3])
    radius_ischemic = 30
    T = 40
v_data_0_1 = compute_v_based_on_reaction_diffusion(
    mesh_file = mesh_file, T = T, submesh_flag = True, ischemia_flag = True,
    gdim = gdim, center_activation = center_activation, radius_activation = radius_activation,
    center_ischemic = center_ischemic, radius_ischemic = radius_ischemic
)

# 0 - 1 to min - max
v_min, v_max = -90, 10
v_data = v_data_0_1 * (v_max - v_min) + v_min

# sample data
# v_data = v_data[::5]
u_data = forward_tmp(mesh_file, v_data, gdim = gdim)
# add snr = 30dB noise
u_data += np.random.normal(0, 0.1, u_data.shape)

if gdim == 2:
    np.save('2d/data/u_data_reaction_diffusion.npy', u_data)
    np.save('2d/data/v_data_reaction_diffusion.npy', v_data)
else:
    np.save('3d/data/u_data_reaction_diffusion.npy', u_data)
    np.save('3d/data/v_data_reaction_diffusion.npy', v_data)