# Using tmp data from ECGsim to compute bsp
# forward : FEM, BEM

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace
from dolfinx.mesh import create_submesh
import h5py
import numpy as np
import scipy.interpolate

from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import compare_bsp_on_standard12lead

ischemia = False
if ischemia:
    file_ecgsim_name = r'forward_inverse_3d/data/ischemia_ECGsim_bem.mat'
else:
    file_ecgsim_name = r'forward_inverse_3d/data/normal_ECGsim_bem.mat'
file_ecgsim = h5py.File(file_ecgsim_name, 'r')
d_data_ecgsim = np.array(file_ecgsim['surface_potential'])

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V2 = functionspace(subdomain, ("Lagrange", 1))

# geom data
geom_data_ecgsim = h5py.File('forward_inverse_3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
v_pts_fem = V2.tabulate_dof_coordinates()

# interpolate ECGsim surface data to FEM volume data
v_data_ecgsim = np.array(file_ecgsim['tmp'])
v_data_fem = []
for timeframe in range(v_data_ecgsim.shape[0]):
    # Interpolate ECGsim data to FEM mesh
    rbf = scipy.interpolate.Rbf(v_pts_ecgsim[:,0], v_pts_ecgsim[:,1], v_pts_ecgsim[:,2], v_data_ecgsim[timeframe])
    v_fem_one = rbf(v_pts_fem[:,0], v_pts_fem[:,1], v_pts_fem[:,2])
    v_data_fem.append(v_fem_one.copy())
v_data_fem = np.array(v_data_fem)
d_data_fem = compute_d_from_tmp(mesh_file, v_data=v_data_fem)

compare_bsp_on_standard12lead(d_data_fem, d_data_ecgsim, 
                              labels = ['FEM', 'BEM'],
                              step_per_timeframe=1,
                              filter_flag=False)