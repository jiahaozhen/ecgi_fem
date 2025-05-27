import sys
import numpy as np
from main_forward_tmp import forward_tmp

sys.path.append(".")
from utils.helper_function import v_data_argument

mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
d_data_file = "3d/data/u_data_reaction_diffusion_ischemia_data_argument.npy"
v_data_file = "3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy"
phi_1_file = "3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
phi_2_file = "3d/data/phi_2_data_reaction_diffusion_ischemia.npy"

v_data = np.load(v_data_file)
d_data = np.load(d_data_file)
phi_1_exact = np.load(phi_1_file)
phi_2_exact = np.load(phi_2_file)

v = v_data_argument(phi_1_exact, phi_2_exact)
d = forward_tmp(mesh_file, v)

print(np.linalg.norm(d - d_data))