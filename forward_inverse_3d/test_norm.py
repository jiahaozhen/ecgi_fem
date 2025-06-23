import sys

from dolfinx.io import gmshio
from dolfinx.mesh import locate_entities_boundary, create_submesh, locate_entities_boundary
from mpi4py import MPI
from dolfinx.plot import vtk_mesh
import pyvista
import numpy as np

sys.path.append('.')
from utils.helper_function import compute_normal

file = "3d/data/mesh_multi_conduct_ecgsim.msh"
# file = "3d/data/heart_torso.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

body_boundary_index = locate_entities_boundary(domain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))
heart_boundary_index = locate_entities_boundary(subdomain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))

nt_f_values = compute_normal(domain)
nh_f_values = compute_normal(subdomain)
nt_check = np.sum(nt_f_values*nt_f_values, axis=1)
nh_check = np.sum(nh_f_values*nh_f_values, axis=1)
indices_nt = np.where(~np.isclose(nt_check, 1))[0]
indices_nh = np.where(~np.isclose(nh_check, 1))[0]
print(nh_check[indices_nh])
print(nt_check[indices_nt])

grid1 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))

plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(grid1, show_edges=True)
plotter.view_xy()
plotter.add_axes()
plotter.add_arrows(domain.geometry.x[body_boundary_index], 
                nt_f_values,
                mag=10,
                color='red')
plotter.subplot(0, 1)
plotter.add_mesh(grid2, show_edges=True)
plotter.view_xy()
plotter.add_axes()
plotter.add_arrows(subdomain.geometry.x[heart_boundary_index], 
                nh_f_values,
                mag=10,
                color='red')
plotter.show()