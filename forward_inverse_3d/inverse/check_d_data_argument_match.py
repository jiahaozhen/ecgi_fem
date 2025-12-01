'''
检查体表电压数据一致性
'''
import numpy as np
from forward_inverse_3d.forward.forward_coupled_ischemia import forward_tmp
from utils.transmembrane_potential_tools import v_data_augment
from utils.visualize_tools import compare_bsp_on_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
d_data_file = r'forward_inverse_3d/data/inverse/u_data_ischemia.npy'
v_data_file = r'forward_inverse_3d/data/inverse/v_data_ischemia.npy'
phi_1_file = r'forward_inverse_3d/data/inverse/phi_1_data_ischemia.npy'
phi_2_file = r'forward_inverse_3d/data/inverse/phi_2_data_ischemia.npy'

v_data = np.load(v_data_file)[0:400]
d_data = np.load(d_data_file)[0:400]
phi_1_exact = np.load(phi_1_file)[0:400]
phi_2_exact = np.load(phi_2_file)[0:400]

v = v_data_augment(phi_1_exact, phi_2_exact)
d, _ = forward_tmp(mesh_file, v)

compare_bsp_on_standard12lead(d_data, d, labels=['no augment', 'augment'], filter_flag=False)