'''
检查体表电压数据一致性
'''
import sys
import numpy as np
from main_forward_tmp import forward_tmp

sys.path.append(".")
from utils.helper_function import v_data_argument
from forward_inverse_3d.forward.forward_coupled import forward_tmp

mesh_file = "forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh"
d_data_file = "forward_inverse_3d/data/u_data_reaction_diffusion_ischemia_data_argument.npy"
v_data_file = "forward_inverse_3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy"
phi_1_file = "forward_inverse_3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
phi_2_file = "forward_inverse_3d/data/phi_2_data_reaction_diffusion_ischemia.npy"

v_data = np.load(v_data_file)
d_data = np.load(d_data_file)
phi_1_exact = np.load(phi_1_file)
phi_2_exact = np.load(phi_2_file)

v = v_data_argument(phi_1_exact, phi_2_exact)
d, _ = forward_tmp(mesh_file, v)

print(np.linalg.norm(d - d_data))