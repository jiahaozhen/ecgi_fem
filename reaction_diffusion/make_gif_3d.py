from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
import numpy as np

mesh_file = '3d/data/mesh_ecgsim_ventricle.msh'
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

u_data = np.load('reaction_diffusion/u_data.npy')
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid["u"] = u_data[0]
plotter.add_mesh(grid, scalars="u", cmap="viridis", clim=[0, 1])
plotter.view_isometric()
plotter.add_text('',name="time_text", font_size=18, color="red")
plotter.open_gif("reaction_diffusion/u_data_3d.gif")

t = 0.0
time_step = 0.2
for i in range(u_data.shape[0]):
    grid["u"] = u_data[i]
    plotter.update_scalars(grid["u"])
    plotter.actors["time_text"].SetText(3, f"Time: {t:.1f} ms")
    plotter.write_frame()
    t += time_step

plotter.show()