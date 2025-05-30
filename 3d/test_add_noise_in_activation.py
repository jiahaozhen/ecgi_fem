import sys

import numpy as np
import multiprocessing

sys.path.append(".")
from utils.helper_function import add_noise_based_on_snr, get_activation_time_from_v, eval_function, compute_phi_with_activation

v_file = '3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy'
v_exact = np.load(v_file)

activation_time_exact = get_activation_time_from_v(v_exact)
activation_time_noise = add_noise_based_on_snr(activation_time_exact, snr=20)

print("Correlation between exact and noisy activation times:", 
      np.corrcoef(activation_time_exact, activation_time_noise)[0, 1])

# plot exact and noisy activation times
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function
from dolfinx.plot import vtk_mesh
import pyvista
mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V = functionspace(subdomain_ventricle, ("Lagrange", 1))
act_f_exact = Function(V)
act_f_exact.x.array[:] = activation_time_exact
act_f_noise = Function(V)
act_f_noise.x.array[:] = activation_time_noise

def plot_f_on_subdomain(f: Function, subdomain, title):
    grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    grid.point_data["f"] = eval_function(f, subdomain.geometry.x)
    grid.set_active_scalars("f")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_title(title)
    plotter.view_yz()
    plotter.add_axes()
    plotter.show()

p1 = multiprocessing.Process(target=plot_f_on_subdomain, args=(act_f_exact, subdomain_ventricle, "Exact Activation Time"))
p2 = multiprocessing.Process(target=plot_f_on_subdomain, args=(act_f_noise, subdomain_ventricle, "Noisy Activation Time"))
p1.start()
p2.start()
p1.join()
p2.join()

# phi_exact = compute_phi_with_activation(act_f_exact, v_exact.shape[0])
phi_noise = compute_phi_with_activation(act_f_noise, v_exact.shape[0])

np.save("3d/data/phi_2_data_reaction_diffusion_ischemia_20dB.npy", phi_noise)