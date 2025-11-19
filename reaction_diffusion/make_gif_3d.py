import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from dolfinx.fem import functionspace, Function
from mpi4py import MPI
import pyvista
import numpy as np
# from main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion
# from main_reaction_diffusion import compute_v_based_on_reaction_diffusion
# from simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion

sys.path.append('.')
from utils.helper_function import eval_function

submesh_flag = True
ischemia_flag = False
gdim = 3
if gdim == 2:
    mesh_file = r'forward_inverse_2d/data/heart_torso.msh'
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    T = 40
    center_activation = np.array([4.0, 4.0])
    radius_activation = 0.1
    center_ischemia = np.array([4.0, 6.0])
    radius_ischemia = 0.5
else:
    if submesh_flag:
        mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
        # mesh of Body
        domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
        tdim = domain.topology.dim
        # mesh of Heart
        subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    else:
        mesh_file = '3d/data/mesh_ecgsim_ventricle.msh'
        subdomain_ventricle, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
        tdim = subdomain_ventricle.topology.dim
    T = 40
    center_activation = np.array([57, 51.2, 15])
    radius_activation = 5
    center_ischemia = np.array([89.1, 40.9, -13.3])
    radius_ischemia = 30

step_per_timeframe = 4
V = functionspace(subdomain_ventricle, ("Lagrange", 1))
u = Function(V)
u_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=ischemia_flag, ischemia_epi_endo=[1, 0, -1], step_per_timeframe=step_per_timeframe)
# u_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=ischemia_flag, surface_flag=True)
# u_data, phi_1, phi_2 = compute_v_based_on_reaction_diffusion(
#     mesh_file, T=T, submesh_flag=submesh_flag, ischemia_flag=ischemia_flag, 
#     gdim=gdim, center_activation=center_activation, radius_activation=radius_activation,
#     center_ischemia=center_ischemia, radius_ischemia=radius_ischemia,
#     data_argument=False, surface_flag=True
# )

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
u.x.array[:] = u_data[0]
grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
plotter.add_mesh(grid, scalars="u", cmap="viridis", clim=[-90, 10])
plotter.view_isometric()
plotter.add_text('',name="time_text", font_size=18, color="red")
if gdim == 2:
    name = "reaction_diffusion/u_data_2d"
else:
    name = "reaction_diffusion/u_data_3d"
if ischemia_flag:
    name += "_ischemia.gif"
else:
    name += "_healthy.gif"
plotter.open_gif(name)

t = 0
time_step = 1 / step_per_timeframe * 1.0  # assuming time unit is ms
for i in range(u_data.shape[0]):
    u.x.array[:] = u_data[i]
    grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
    plotter.update_scalars(grid["u"])
    plotter.actors["time_text"].SetText(3, f"Time: {t:.1f} ms")
    plotter.write_frame()
    t += time_step

plotter.close()