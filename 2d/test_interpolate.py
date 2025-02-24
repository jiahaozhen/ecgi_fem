from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from mpi4py import MPI
import pyvista
import matplotlib.pyplot as plt

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

V2 = functionspace(subdomain, ("Lagrange", 1))

# phi exact
class exact_solution():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2
        return dist - self.r ** 2

# preparation 
a1 = -60
a2 = -90
tau = 1
phi_exact = exact_solution(4, 2, 1)
phi = Function(V2)
# phi.interpolate(phi_exact)
x = 6
y = 4
r = 1
phi.x.array[:] = np.sqrt((subdomain.geometry.x[:,0]-x)**2 + (subdomain.geometry.x[:,1]-y)**2)-r

marker = np.zeros(subdomain.topology.index_map(tdim-2).size_local)
marker[phi.x.array < 0] = 1
grid0 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid0.point_data["phi"] = phi.x.array
grid0.set_active_scalars("phi")
grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid1.point_data["marker"] = marker
grid1.set_active_scalars("marker")

plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(grid0, show_edges=True)
plotter.view_xy()
plotter.add_axes()
plotter.subplot(0, 1)
plotter.add_mesh(grid1, show_edges=True)
plotter.view_xy()
plotter.add_axes()
plotter.show()