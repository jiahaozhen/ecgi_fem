# Using tmp data from ECGsim to compute bsp
# forward : FEM, BEM
# save the result in .mat format

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace
from dolfinx.mesh import locate_entities_boundary, create_submesh
import h5py
import scipy.io as sio
import numpy as np
import scipy.interpolate
from forward_inverse_3d.forward.forward_coupled_ischemia import compute_d_from_tmp
from utils.helper_function import transfer_bsp_to_standard12lead
from utils.visualize_tools import compare_standard_12_lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# geom data
geom_data_ecgsim = h5py.File('forward_inverse_3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
d_pts_ecgsim = np.array(geom_data_ecgsim['geom_thorax']['pts'])
v_pts_fem = V2.tabulate_dof_coordinates()
d_index = locate_entities_boundary(domain, tdim - 3, 
                                   lambda x: np.full(x.shape[1], True, dtype=bool))
d_pts_fem = V1.tabulate_dof_coordinates()[d_index]

ischemia = False
if ischemia:
    file_ecgsim_name = r'forward_inverse_3d/data/ischemia_ECGsim_bem.mat'
else:
    file_ecgsim_name = r'forward_inverse_3d/data/normal_ECGsim_bem.mat'
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
d_data_fem = compute_d_from_tmp(mesh_file, v_data=v_data_fem, ischemia_flag=ischemia)

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_fem = transfer_bsp_to_standard12lead(d_data_fem, leadIndex)
standard12Lead_bem = transfer_bsp_to_standard12lead(d_data_ecgsim, leadIndex)

compare_standard_12_lead(standard12Lead_fem, standard12Lead_bem, 
                         labels = ['FEM', 'BEM'],
                         step_per_timeframe=1,
                         filter_flag=False)

# if ischemia:
#     sio.savemat('3d/data/ischemia_ECGsim_fem.mat', {'surface_potential': d_data_fem})
# else:
#     sio.savemat('3d/data/normal_ECGsim_fem.mat', {'surface_potential': d_data_fem})