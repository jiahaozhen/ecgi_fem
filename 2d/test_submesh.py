from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import pyvista
import numpy as np

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V2 = functionspace(subdomain, ("Lagrange", 1))
mesh = subdomain

a1 = -45
a2 = -85
num_node = mesh.topology.index_map(tdim-2).size_local
# phi exact
class exact_solution():
    def __init__(self, x, r):
        self.xi = x
        self.r = r
    def __call__(self, x):
        dist = (x[0]-self.xi[0])**2 + (x[1]-self.xi[1])**2
        dist = np.sqrt(dist)
        return dist - self.r
        
phi_exact = exact_solution([4,6], 1)
phi = Function(V2)
phi.interpolate(phi_exact)

tdim = mesh.topology.dim
p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, tdim))
grid.point_data["Marker"] = phi.vector.array
grid.set_active_scalars("Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("domains_structured.png")
p.show()