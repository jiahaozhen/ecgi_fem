import scipy.io as sio
import h5py
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace
from dolfinx.mesh import create_submesh
import scipy.interpolate

from utils.error_metrics_tools import compute_error_and_correlation
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp

ischemia = True

if ischemia:
    file_ecgsim = h5py.File(r'forward_inverse_3d/data/ischemia_ECGsim_bem.mat', 'r')
else:
    file_ecgsim = h5py.File(r'forward_inverse_3d/data/normal_ECGsim_bem.mat', 'r')

d_data_ecgsim = np.array(file_ecgsim['surface_potential'])
v_data_ecgsim = np.array(file_ecgsim['tmp'])

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V2 = functionspace(subdomain, ("Lagrange", 1))

# geom data
geom_data_ecgsim = h5py.File('forward_inverse_3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
v_pts_fem = V2.tabulate_dof_coordinates()

v_data_fem = []
for timeframe in range(v_data_ecgsim.shape[0]):
    # Interpolate ECGsim data to FEM mesh
    rbf = scipy.interpolate.Rbf(v_pts_ecgsim[:,0], v_pts_ecgsim[:,1], v_pts_ecgsim[:,2], v_data_ecgsim[timeframe])
    v_fem_one = rbf(v_pts_fem[:,0], v_pts_fem[:,1], v_pts_fem[:,2])
    v_data_fem.append(v_fem_one.copy())
v_data_fem = np.array(v_data_fem)

sigma_i = 1
sigma_e_range = range(0, 20)
sigma_t_range = range(0, 20)
re_table = np.zeros((len(sigma_e_range),len(sigma_t_range)), dtype=float)
cc_table = np.zeros((len(sigma_e_range),len(sigma_t_range)), dtype=float)

index_e = 0
index_t = 0
re_min = 1
# best match so far : sigma_e = 1.7 sigma_t = 2.6 re: 0.1325
for e in sigma_e_range:
    sigma_e = (e/10 + 1) * sigma_i
    for t in sigma_t_range:
        sigma_t = (t/10 + 1) * sigma_i
        d_data_fem = compute_d_from_tmp(mesh_file,
                                        v_data=v_data_fem,
                                        sigma_i=sigma_i, 
                                        sigma_e=sigma_e, 
                                        sigma_t=sigma_t).squeeze()
        re, cc = compute_error_and_correlation(d_data_fem, d_data_ecgsim)
        if re < re_min:
            index_e = e
            index_t = t
            re_min = re
        re_table[e, t] = re
        cc_table[e, t] = cc
print('best choice for sigma is: sigma_e:',index_e/10+1, ',sigma_t:',index_t/10+1, 're:', re_min)