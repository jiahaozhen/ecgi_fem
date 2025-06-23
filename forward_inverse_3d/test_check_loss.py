import sys
import time

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from main_forward_tmp import forward_tmp
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, eval_function, compute_phi_with_v_timebased

# in forward problem, u based on v is recommended
# loss is too big to be optimized
gdim = 3
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
domain_boundary = locate_entities_boundary(domain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

# paramter
a1 = -90 # no active no ischemia
a2 = -60 # no active ischemia
a3 = 10  # active no ischemia
a4 = -20 # active ischemia
tau = 5

# phi phi_est G_phi delta_phi delta_deri_phi
phi_1 = Function(V2)
phi_2 = Function(V2)
G_phi_1 = Function(V2)
G_phi_2 = Function(V2)

# function u w d
u = Function(V1)
d = Function(V1)
v = Function(V2)
# define d's value on the boundary
u_exact_all_time = np.load(d_data_file)
time_total = np.shape(u_exact_all_time)[0]

# scalar c
ds = Measure('ds', domain = domain)
c1_element = (d - u) * ds
c2_element = 1 * ds
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element = 0.5 * (u - d) ** 2 * ds
form_loss = form(loss_element)

form_u = form(0.5 * u ** 2 * ds)

# exact v
v_exact_all_time = np.load(v_exact_data_file)
v_result_all_time = np.full_like(v_exact_all_time, 0.0)

# exact phi_1 phi_2
phi_1_exact_all_time, phi_2_exact_all_time = compute_phi_with_v_timebased(v_exact_all_time, V2, a2, a4)

u_result_all_time = np.full_like(u_exact_all_time, 0.0)
for i in range(time_total):
    # u based v from phi_1 phi_2
    phi_1.x.array[:] = phi_1_exact_all_time[i]
    phi_2.x.array[:] = phi_2_exact_all_time[i]
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    v_result_all_time[i] = (a1*G_phi_2.x.array + a3*(1-G_phi_2.x.array)) * G_phi_1.x.array + (a2*G_phi_2.x.array + a4*(1-G_phi_2.x.array)) * (1-G_phi_1.x.array)
u_result_on_v_all_time = forward_tmp(mesh_file, v_result_all_time, gdim = gdim)
# u_exact_all_time = forward_tmp(mesh_file, v_exact_all_time, gdim = gdim)

plt.subplot(1, 2, 1)
plt.plot(v_exact_all_time[:,100])
plt.subplot(1, 2, 2)
plt.plot(v_result_all_time[:,100])
plt.show()

cc = []
loss_list = []
for i in range(u_exact_all_time.shape[0]):
    u.x.array[:] = u_result_on_v_all_time[i]
    # u.x.array[:] = u_exact_all_time[i] + 0.1 * np.random.randn(u_exact_all_time[i].shape[0])
    d.x.array[:] = u_exact_all_time[i]
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u_result_on_v_all_time[i] = u.x.array + adjustment
    u.x.array[:] = u.x.array + adjustment
    loss = assemble_scalar(form_loss) / assemble_scalar(form_c2)
    loss_list.append(loss)
    # cc.append(np.corrcoef(u.x.array, d.x.array)[0, 1])
    cc.append(np.corrcoef(v_result_all_time[i], v_exact_all_time[i])[0, 1])
cc = np.array(cc)
loss_list = np.array(loss_list)
plt.subplot(1, 2, 1)
plt.plot(cc)
plt.subplot(1, 2, 2)
plt.plot(loss_list)
plt.show()

def plot_d_on_surface(u_data):
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    u.x.array[:] = u_data
    u_boundary = eval_function(u, domain.geometry.x).squeeze()
    mask = ~np.isin(np.arange(len(u_boundary)), domain_boundary)
    u_boundary[mask] = u_boundary[domain_boundary[0]]
    grid.point_data['u'] = u_boundary
    grid.set_active_scalars('u')
    plotter.add_mesh(grid, show_edges=True)
    plotter.show()

p1 = multiprocessing.Process(target=plot_d_on_surface, args=(u_result_on_v_all_time[15],))
p2 = multiprocessing.Process(target=plot_d_on_surface, args=(u_exact_all_time[15],))
p3 = multiprocessing.Process(target=plot_d_on_surface, args=(u_exact_all_time[100] - u_result_on_v_all_time[100],))
# p1.start()
# p2.start()
# p3.start()
# p1.join()
# p2.join()
# p3.join()