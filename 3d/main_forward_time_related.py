import sys

import numpy as np
from main_forward_tmp import forward_tmp
from main_create_mesh_ecgsim_multi_conduct import create_mesh

sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
create_mesh(mesh_file, lc = 40)
v_data_0_1 = compute_v_based_on_reaction_diffusion(
    mesh_file, T = 50, submesh_flag = True, ischemia_flag = True)
# 0 - 1 to min - max
v_min, v_max = -90, 10
v_data = v_data_0_1 * (v_max - v_min) + v_min

u_data = forward_tmp(mesh_file, v_data)

np.save('3d/data/u_data_reaction_diffusion.npy', u_data)
np.save('3d/data/v_data_reaction_diffusion.npy', v_data)