import scipy.io as sio

from forward_inverse_3d.mesh.create_mesh_ecgsim_multi_conduct import create_mesh
from forward_inverse_3d.deprecated.main_ecgsim2fem import ecgsim2fem
from forward_inverse_3d.main_forward_tmp import compute_d_from_tmp

file = '3d/data/mesh_multi_conduct_ecgsim.msh'
for lc in range(10, 101, 5):
    create_mesh(file, lc, False)
    ecgsim2fem(file)
    d_all = compute_d_from_tmp(file, multi_flag=False)
    sio.savemat('3d/data/surface_potential_fem_two_conduct_lc_' + str(lc) + '.mat', {'surface_potential_fem': d_all})
    print('lc:', lc, 'done')

for lc in range(10, 101, 5):
    create_mesh(file, lc, True)
    ecgsim2fem(file)
    d_all = compute_d_from_tmp(file, multi_flag=True)
    sio.savemat('3d/data/surface_potential_fem_multi_conduct_lc_' + str(lc) + '.mat', {'surface_potential_fem': d_all})
    print('lc:', lc, 'done')

lc = None
create_mesh(file, lc, True)
ecgsim2fem(file)
d_all = compute_d_from_tmp(file, multi_flag=True)
sio.savemat('3d/data/surface_potential_fem_multi_conduct_lc_None.mat', {'surface_potential_fem': d_all})