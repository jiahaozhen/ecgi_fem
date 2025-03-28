import sys

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
import h5py
import numpy as np
import scipy.io as sio

sys.path.append('.')
from utils.helper_function import eval_function

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

u = Function(V1)
v = Function(V2)

v_data_fem = np.load('3d/data/v_data_reaction_diffusion.npy')
geom_data_ecgsim = h5py.File('3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
d_pts_ecgsim = np.array(geom_data_ecgsim['geom_thorax']['pts'])

t = v_data_fem.shape[0]
print(t)
v_data_ecgsim = np.zeros((t, v_pts_ecgsim.shape[0]))
for timeframe in range(t):
    v.x.array[:] = v_data_fem[timeframe]
    v_data_ecgsim[timeframe] = eval_function(v, v_pts_ecgsim).reshape(-1)

forward_matrix_matrix = h5py.File('3d/data/forward_matrix_ecgsim.mat', 'r')
forward_matrix = np.array(forward_matrix_matrix['forward_matrix'])

ecg_ecgsim = v_data_ecgsim @ forward_matrix
ecg_fem_origin = np.load('3d/data/u_data_reaction_diffusion.npy')

ecg_fem = np.zeros((t, ecg_ecgsim.shape[1]))
for timeframe in range(t):
    u.x.array[:] = ecg_fem_origin[timeframe]
    ecg_fem[timeframe] = eval_function(u, d_pts_ecgsim).reshape(-1)

ecg_fem = ecg_fem + np.mean(ecg_ecgsim - ecg_fem, axis=1, keepdims=True)
# print(np.mean(ecg_ecgsim - ecg_fem, axis=1))

# cc of every time frame
cc = np.zeros(t)
for i in range(t):
    cc[i] = np.corrcoef(ecg_ecgsim[i], ecg_fem[i])[0, 1]
print(np.mean(cc))

# re of every time frame
re = np.zeros(t)
for i in range(t):
    re[i] = np.linalg.norm(ecg_ecgsim[i] - ecg_fem[i]) / np.linalg.norm(ecg_ecgsim[i])
print(np.mean(re))

sio.savemat('3d/data/reaction_diffusion_ischemia_result.mat', {'v_data_ecgsim': v_data_ecgsim, 'ecg_ecgsim': ecg_ecgsim, 'ecg_fem': ecg_fem, 'cc': cc, 're': re})
