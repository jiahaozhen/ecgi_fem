import sys
sys.path.append('.')
from utils.helper_function import eval_function

from main_forward_tmp import forward_tmp, extract_d_from_u
import numpy as np
import h5py
import scipy.io as sio
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
import numpy as np
from mpi4py import MPI

mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

v_fem = np.load('3d/data/v_data_reaction_diffusion.npy')
u_fem = np.load('3d/data/u_data_reaction_diffusion.npy')
geom = h5py.File('3d/data/geom_ecgsim.mat', 'r')
points_thorax = np.array(geom['geom_thorax']['pts'])
points_ventricle = np.array(geom['geom_ventricle']['pts'])
d_fem = extract_d_from_u(mesh_file, points_thorax, u_fem)

V = functionspace(subdomain_ventricle, ("Lagrange", 1))
v = Function(V)
v_ecgsim = []
total_num = len(v_fem)
for i in range(total_num):
    v.x.array[:] = v_fem[i]
    v_surface = eval_function(v, points=points_ventricle)
    v_ecgsim.append(v_surface.copy())
v_ecgsim = np.array(v_ecgsim)

# surface_potential_file = '3d/data/reaction_diffusion_ischemia.mat'
surface_potential_file = '3d/data/reaction_diffusion_normal.mat'
sio.savemat(surface_potential_file, {'d_fem': d_fem, 'v_fem': v_fem, 'v_ecgsim': v_ecgsim})