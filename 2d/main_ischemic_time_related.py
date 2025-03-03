import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
from helper_function import compare_CM
import numpy as np
import pyvista
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, eval_function

# mesh of Body
file = "2d/data/heart_torso.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(0).size_local
sub_boundary_index = locate_entities_boundary(subdomain, tdim-2, lambda x: np.full(x.shape[1], True, dtype=bool))

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# sigma_i : intra-cellular conductivity tensor in Heart
# sigma_e : extra-cellular conductivity tensor in Heart
# sigma_t : conductivity tensor in Torso
# M  : sigma_i + sigma_e in Heart 
#      sigma_t in Torso
sigma_t = 0.8
sigma_i = 0.4
sigma_e = 0.8

def rho1(x):
    tensor = np.eye(tdim) * sigma_t
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(tdim) * (sigma_i + sigma_e)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
Mi = Constant(subdomain, default_scalar_type(np.eye(tdim)*sigma_i))

# parameter a1 a2 a3 a4 tau
a1 = -90 # no active no ischemia
a2 = -60 # no active ischemia
a3 = 10  # active no ischemia
a4 = -20 # active ischemia
tau = 0.05
alpha1 = 1e-1
alpha2 = 1e-5

# phi G_phi delta_phi delta_deri_phi
phi_1 = Function(V2)
phi_2 = Function(V2)
phi_1_est = Function(V2)
phi_2_est = Function(V2)
G_phi_1 = Function(V2)
G_phi_2 = Function(V2)
delta_phi_1 = Function(V2)
delta_phi_2 = Function(V2)
delta_deri_phi_1 = Function(V2)
delta_deri_phi_2 = Function(V2)

u = Function(V1)
w = Function(V1)
# function d
d = Function(V1)
# define d's value on the boundary
d_all_time = np.load(file='2d/data/bsp_all_time.npy')
time_total = np.shape(d_all_time)[0]

# matrix A_u
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx", domain = domain)
a_u = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form_a = form(a_u)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()

solver = PETSc.KSP().create()
solver.setOperators(A_u)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.HYPRE)

# vector b_u
dx2 = Measure("dx", domain = subdomain)
b_u_element = -(a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 \
        - (a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2 \
        - (a3 - a4) * delta_phi_1 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 \
        - (a2 - a4) * delta_phi_2 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b_u = form(b_u_element, entity_maps = entity_map)
b_u = create_vector(linear_form_b_u)

# scalar c
ds = Measure('ds', domain = domain)
c1_element = (d-u) * ds
c2_element = 1 * ds
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element_1 = 0.5 * (u - d) ** 2 * ds
loss_element_2 = 0.5 * alpha1 * (phi_1 - phi_1_est) ** 2 * dx2 \
    + 0.5 * alpha2 * (phi_2 - phi_2_est) ** 2 * dx2 \
#     + 0.5 * alpha1 * dot(grad(phi_1 - phi_1_prior), grad(phi_1 - phi_1_prior)) * dx2 \
#     + 0.5 * alpha2 * dot(grad(phi_2 - phi_2_prior), grad(phi_2 - phi_2_prior)) * dx2
form_loss_1 = form(loss_element_1)
form_loss_2 = form(loss_element_2)

# vector b_w
b_w_element = u1 * (u - d) * ds
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction p
u2 = TestFunction(V2)
j_p = -(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a1 - a2) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a1 - a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        - (a3 - a4) * delta_deri_phi_1 * (1 - G_phi_2) * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a3 - a4) * delta_phi_1 * (1 - G_phi_2) * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        + alpha1 * (phi_1 - phi_1_est) * u2 * dx2
form_J_p = form(j_p, entity_maps=entity_map)
J_p = create_vector(form_J_p)

# vector direction q
j_q = -(a1 - a2 -a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a1 - a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a1 - a3) * delta_phi_2 * G_phi_1 * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        - (a2 - a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a2 - a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        + alpha2 * (phi_2 - phi_2_est) * u2 * dx2
form_J_q = form(j_q, entity_maps=entity_map)
J_q = create_vector(form_J_q)

# exact phi_1 phi_2
phi_1_exact = Function(V2)
phi_1_exact.x.array[:] = np.load(file='2d/data/phi_1.npy')
phi_2_exact_all_time = np.load('2d/data/phi_2_all_time.npy')

# initial phi_1 phi_2
phi_0 = np.full(phi_1.x.array.shape, tau/2)
phi_1.x.array[:] = phi_0
phi_2.x.array[:] = phi_0

# phi result
phi_1_result = np.zeros((time_total, sub_node_num))
phi_2_result = np.zeros((time_total, sub_node_num))

cm_phi_1_per_timeframe = []
cm_phi_2_per_timeframe = []
loss_per_timeframe = []

for timeframe in range(time_total):

    # define d's value on the boundary
    d.x.array[:] = d_all_time[timeframe]
    # phi_1.x.array[:] = phi_1_exact.x.array
    # phi_2.x.array[:] = phi_2_exact_all_time[timeframe]
    if timeframe == 0:
        phi_1_est.x.array[:] = phi_1.x.array
        phi_2_est.x.array[:] = phi_2.x.array
    elif timeframe == 1:
        phi_1_est.x.array[:] = phi_1_result[timeframe-1]
        phi_2_est.x.array[:] = phi_2_result[timeframe-1]
    else:
        phi_1_est.x.array[:] = np.mean(phi_1_result[0:timeframe], axis=0)
        phi_2_est.x.array[:] = 2 * phi_2_result[timeframe-1] - phi_2_result[timeframe-2]

    # phi_1.x.array[:] = phi_0
    # phi_2.x.array[:] = phi_0

    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
    delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
    delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)

    # get u from p, q
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1)/assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    k = 0
    while (k < 1e2):
       # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)
        # compute partial derivative of p q
        with J_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_p, form_J_p)
        with J_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_q, form_J_q)
        # cost function
        loss = assemble_scalar(form_loss_1)\
             + assemble_scalar(form_loss_2)

        # check if the condition is satisfie
        if (np.linalg.norm(J_p.array) < 1e-1 and np.linalg.norm(J_q.array) < 1e-1):
            break
        # if (np.linalg.norm(J_p.array) < 1e-2 and loss < 1e-2):
            # break
        
        # line search
        phi_1_v = phi_1.x.array[:].copy()
        phi_2_v = phi_2.x.array[:].copy()
        J_p_array = J_p.array.copy()
        J_q_array = J_q.array.copy()
        J_p_array[sub_boundary_index] = J_p_array[sub_boundary_index] / 10
        J_q_array[sub_boundary_index] = J_q_array[sub_boundary_index] / 10
        alpha = 1
        gamma = 0.6
        c = 0.1
        while(True):
            # adjust p q
            phi_1.x.array[:] = phi_1_v - alpha * J_p_array
            phi_2.x.array[:] = phi_2_v - alpha * J_q_array
            # get u from p, q
            G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
            delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment
            # compute loss
            loss_new = assemble_scalar(form_loss_1) \
                + assemble_scalar(form_loss_2)
            loss_cmp = loss_new - (loss - c * alpha * np.linalg.norm(np.concatenate((J_p.array, J_q.array)))**2)
            if (loss_cmp < 1e-2):
                break
            alpha = gamma * alpha
        k = k + 1

    print('timeframe:', timeframe)
    print('end at', k, 'iteration')
    print('J_p:', np.linalg.norm(J_p.array))
    print('J_q:', np.linalg.norm(J_q.array))
    print('loss:', loss)
    phi_1_result[timeframe] = phi_1.x.array
    phi_2_result[timeframe] = phi_2.x.array
    cm_phi_1_per_timeframe.append(compare_CM(subdomain, phi_1_exact, phi_1))
    loss_per_timeframe.append(loss)

# np.save('2d/data/phi_1_result.npy', phi_1.x.array)
# np.save('2d/data/phi_2_result.npy', phi_2.x.array)
# np.save('2d/data/phi_1_exact.npy', phi_1_exact.x.array)

# check result
marker_exact = np.zeros(sub_node_num)
marker_exact[phi_1_exact.x.array < 0] = 1
marker_result = np.zeros(sub_node_num)
marker_result[np.mean(phi_1_result, axis=0) < 0] = 1

# phi_1_average = Function(V2)
# phi_1_average.x.array[:] = np.mean(phi_1_result, axis=0)
# print(phi_1_average.x.array)
# print(compare_CM(subdomain, phi_1_exact, phi_1_average))

plt.subplot(1, 2, 1)
plt.plot(cm_phi_1_per_timeframe)
plt.title('error in center of mass')
plt.xlabel('time')
plt.subplot(1, 2, 2)
plt.plot(loss_per_timeframe)
plt.title('cost functional')
plt.xlabel('time')
plt.show()

marker = Function(V2)
grid0 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
marker.x.array[:] = marker_exact
grid0.point_data["marker_exact"] = eval_function(marker, subdomain.geometry.x)
grid0.set_active_scalars("marker_exact")
grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
marker.x.array[:] = marker_result
grid1.point_data["marker_reult"] = eval_function(marker, subdomain.geometry.x)
grid1.set_active_scalars("marker_reult")
# grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
# grid2.point_data["phi_1_exact"] = eval_function(phi_1_exact, subdomain.geometry.x)
# grid2.set_active_scalars("phi_1_exact")
# grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
# grid3.point_data["phi_1_result"] = eval_function(phi_1, subdomain.geometry.x)
# grid3.set_active_scalars("phi_1_result")
# grid4 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
# marker.x.array[:] = phi_2_exact_all_time[time_total-1]
# grid4.point_data["phi_2_exact"] = eval_function(marker, subdomain.geometry.x)
# grid4.set_active_scalars("phi_2_exact")
# grid5 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
# grid5.point_data["phi_2_result"] = eval_function(phi_2, subdomain.geometry.x)
# grid5.set_active_scalars("phi_2_result")

# grid = (grid0, grid2, grid4, grid1, grid3, grid5)

grid = (grid0, grid1)
plotter = pyvista.Plotter(shape=(1, 2))
for i in range(1):
    for j in range(2):
        plotter.subplot(i, j)
        plotter.add_mesh(grid[i+j], show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
plotter.show()

# plotter = pyvista.Plotter(shape=(2, 4))
# for i in range(2):
#     for j in range(4):
#         plotter.subplot(i, j)
#         grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
#         grid.point_data["phi_2"] = phi_2_result[i*20 + j*5]
#         grid.set_active_scalars("phi_2")
#         # grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
#         # grid.point_data["u"] = u_all_time[i*20 + j*5]
#         # grid.set_active_scalars("u")
#         plotter.add_mesh(grid, show_edges=True)
#         plotter.view_xy()
#         plotter.add_axes()
# plotter.show()