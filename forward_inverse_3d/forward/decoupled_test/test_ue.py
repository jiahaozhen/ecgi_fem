import sys

import numpy as np

sys.path.append('.')
from utils.function_tools import extract_data_from_function
from forward_inverse_3d.forward.forward_coupled import forward_tmp
from forward_inverse_3d.forward.forward_decoupled import forward_tmp2ue
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3
T = 1
step_per_timeframe = 2
v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T, step_per_timeframe=step_per_timeframe)

# method 1
ue_f_data, ue_functionspace = forward_tmp2ue(mesh_file, v_data[0], gdim=gdim)
mesh_pts = ue_functionspace.mesh.geometry.x
ue_mesh_data_0 = extract_data_from_function(ue_f_data, ue_functionspace, mesh_pts)

# method 2
u_f_data, u_functionspace = forward_tmp(mesh_file, v_data[0], gdim=gdim)
ue_mesh_data_1 = extract_data_from_function(u_f_data, u_functionspace, mesh_pts)

diff = ue_mesh_data_0 - ue_mesh_data_1
print("Max difference between two methods for ue at mesh points:", np.max(np.abs(diff)))
print("Mean difference between two methods for ue at mesh points:", np.mean(np.abs(diff)))
print("Min difference between two methods for ue at mesh points:", np.min(np.abs(diff)))