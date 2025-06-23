from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
import numpy as np
import pyvista

mesh, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)

tdim = mesh.topology.dim
p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, tdim))
# grid.cell_data["Marker"] = np.arange(num_cell)
grid.cell_data["Marker"] = cell_markers.values
grid.set_active_scalars("Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("domains_structured.png")
p.show()