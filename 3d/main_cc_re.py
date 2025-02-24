import sys

import h5py
import numpy as np

sys.path.append('.')
from utils.helper_function import compute_error_and_correlation
from main_create_mesh_ecgsim_multi_conduct import create_mesh
from main_ecgsim2fem import ecgsim2fem
from main_forward_tmp import compute_d_from_tmp

ischemic = False
multi_flag = True
file = '3d/data/mesh_multi_conduct_ecgsim.msh'
create_mesh(file, None, multi_flag=multi_flag)
ecgsim2fem(file, ischemic=ischemic)
if ischemic:
    file_ecgsim = h5py.File('3d/data/ischemic_ecgsim.mat', 'r')
else:
    file_ecgsim = h5py.File('3d/data/sinus_rhythm_ecgsim.mat', 'r')
d_data_ecgsim = np.array(file_ecgsim['surface_potential'])

d_data_fem = compute_d_from_tmp(file, multi_flag=multi_flag).squeeze()
np.save('3d/data/surface_potential_fem.npy', d_data_fem)
print(compute_error_and_correlation(d_data_fem, d_data_ecgsim))