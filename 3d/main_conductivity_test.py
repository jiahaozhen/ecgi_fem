import sys

import scipy.io as sio
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
sigma_i = 1
sigma_e_range = range(0, 20)
sigma_t_range = range(0, 20)
re_table = np.zeros((len(sigma_e_range),len(sigma_t_range)), dtype=float)
cc_table = np.zeros((len(sigma_e_range),len(sigma_t_range)), dtype=float)

index_e = None
index_t = None
re_min = 1
# best match so far : sigma_e = 1.7 sigma_t = 2.6 re: 0.1325
for e in sigma_e_range:
    sigma_e = (e/10 + 1) * sigma_i
    for t in sigma_t_range:
        sigma_t = (t/10 + 1) * sigma_i
        d_data_fem = compute_d_from_tmp(file, multi_flag=multi_flag,
                                         sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t).squeeze()
        re, cc = compute_error_and_correlation(d_data_fem, d_data_ecgsim)
        if re < re_min:
            index_e = e
            index_t = t
            re_min = re
        re_table[e, t] = re
        cc_table[e, t] = cc
print('best choice for sigma is: sigma_e:',index_e/10+1, ',sigma_t:',index_t/10+1, 're:', re_min)
# print(re, cc)
# np.save('3d/data/re_table_conductivity.npy', re_table)
# np.save('3d/data/cc_table_conductivity.npy', cc_table)
sio.savemat('3d/data/re_cc_conductivity.mat', {'re': re_table, 'cc': cc_table})