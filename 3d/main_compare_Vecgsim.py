# Using tmp data from ECGsim to compute bsp
# forward : FEM, BEM
# save the result in .mat format
import sys

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace
from dolfinx.mesh import locate_entities_boundary, create_submesh
import h5py
import scipy.io as sio
import numpy as np
import scipy.interpolate

sys.path.append('.')
from main_forward_tmp import compute_d_from_tmp

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# geom data
geom_data_ecgsim = h5py.File('3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
d_pts_ecgsim = np.array(geom_data_ecgsim['geom_thorax']['pts'])
v_pts_fem = V2.tabulate_dof_coordinates()
d_index = locate_entities_boundary(domain, tdim - 3, 
                                   lambda x: np.full(x.shape[1], True, dtype=bool))
d_pts_fem = V1.tabulate_dof_coordinates()[d_index]

ischemia = True
if ischemia:
    file_ecgsim_name = '3d/data/ischemia_ECGsim_bem.mat'
else:
    file_ecgsim_name = '3d/data/normal_ECGsim_bem.mat'
file_ecgsim = h5py.File(file_ecgsim_name, 'r')
v_data_ecgsim = np.array(file_ecgsim['tmp'])
d_data_ecgsim = np.array(file_ecgsim['surface_potential'])

# interpolate ECGsim surface data to FEM volume data
v_data_fem = []
for timeframe in range(v_data_ecgsim.shape[0]):
    # Interpolate ECGsim data to FEM mesh
    rbf = scipy.interpolate.Rbf(v_pts_ecgsim[:,0], v_pts_ecgsim[:,1], v_pts_ecgsim[:,2], v_data_ecgsim[timeframe])
    v_fem_one = rbf(v_pts_fem[:,0], v_pts_fem[:,1], v_pts_fem[:,2])
    v_data_fem.append(v_fem_one.copy())
v_data_fem = np.array(v_data_fem)
d_data_fem = compute_d_from_tmp(mesh_file, v_data=v_data_fem)

if ischemia:
    sio.savemat('3d/data/ischemia_ECGsim_fem.mat', {'surface_potential': d_data_fem})
else:
    sio.savemat('3d/data/normal_ECGsim_fem.mat', {'surface_potential': d_data_fem})