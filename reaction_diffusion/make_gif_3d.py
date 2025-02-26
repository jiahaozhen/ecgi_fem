import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from dolfinx.fem import functionspace, Function
from mpi4py import MPI
import pyvista
from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion

sys.path.append('.')
from utils.helper_function import eval_function

submesh_flag = False
if submesh_flag:
    mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
else:
    mesh_file = '3d/data/mesh_ecgsim_ventricle.msh'
    subdomain_ventricle, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    tdim = subdomain_ventricle.topology.dim

V = functionspace(subdomain_ventricle, ("Lagrange", 1))
u = Function(V)
u_data = compute_v_based_on_reaction_diffusion(mesh_file)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
u.x.array[:] = u_data[0]
grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
plotter.add_mesh(grid, scalars="u", cmap="viridis", clim=[0, 1])
plotter.view_isometric()
plotter.add_text('',name="time_text", font_size=18, color="red")
plotter.open_gif("reaction_diffusion/u_data_3d.gif")

t = 0.0
time_step = 0.2
for i in range(u_data.shape[0]):
    u.x.array[:] = u_data[i]
    grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
    plotter.update_scalars(grid["u"])
    plotter.actors["time_text"].SetText(3, f"Time: {t:.1f} ms")
    plotter.write_frame()
    t += time_step

plotter.show()