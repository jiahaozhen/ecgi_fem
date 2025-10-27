'''
边界赋值测试
'''
import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import numpy as np
import pyvista

sys.path.append('.')
from utils.helper_function import eval_function, assign_function

gdim = 2
if gdim == 2:
    mesh_file = '2d/data/heart_torso.msh'
    v_exact_data_file = '2d/data/v_data_reaction_diffusion.npy'
    d_data_file = '2d/data/u_data_reaction_diffusion.npy'
else:
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    v_exact_data_file = '3d/data/v_data_reaction_diffusion.npy'
    d_data_file = '3d/data/u_data_reaction_diffusion.npy'

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim = gdim)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local
sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

v = Function(V2)
v_value = np.zeros((sub_node_num))
v_value[sub_domain_boundary] = 1
assign_function(v, np.arange(len(v_value)), v_value)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data['v'] = eval_function(v, subdomain_ventricle.geometry.x)
grid.set_active_scalars('v')
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()