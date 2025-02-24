from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary, meshtags
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils.helper_function import delta_tau, delta_deri_tau, compute_error, petsc2array, eval_function

# mesh of Body
file = "2d/data/heart_torso.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(0).size_local

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart 
#      M0 in Torso

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
a3 = 20 # active no ischemia
a4 = -10 # active ischemia
tau = 0.3
# alpha1 = 1e-3
# alpha2 = 1e-5

# phi G_phi delta_phi delta_deri_phi
phi_1 = Function(V2)
phi_2 = Function(V2)
# phi_1_prior = Function(V2)
# phi_2_prior = Function(V2)
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
dx1 = Measure("dx", domain=domain)
a_u = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form_a = form(a_u)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()

solver = PETSc.KSP().create()
solver.setOperators(A_u)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# vector b_u
dx2 = Measure("dx", domain=subdomain)
b_u_element = (a2-a3) * delta_phi_1 * G_phi_2 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 + \
        (a2-a3) * delta_phi_2 * G_phi_1 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2 + \
        (a1-a2) * delta_phi_2 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b_u = form(b_u_element, entity_maps=entity_map)
b_u = create_vector(linear_form_b_u)

# outerBoundary
facets = locate_entities_boundary(domain, tdim - 1, OuterBoundary1)
facet_indices = np.array(facets, dtype=np.int32)
facet_values = np.full(len(facet_indices), 1, dtype=np.int32)
facet_tags = meshtags(domain, tdim - 1, facet_indices, facet_values)
ds_out = Measure('ds', domain=domain, subdomain_data=facet_tags, subdomain_id=1)

# scalar c
c1_element = (d-u) * ds_out
c2_element = 1 * ds_out
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element_1 = 0.5 * (u - d) ** 2 * ds_out
loss_element_2 = 0.5 * alpha1 * (phi_1 - phi_1_prior) ** 2 * dx2 \
    + 0.5 * alpha2 * (phi_2 - phi_2_prior) ** 2 * dx2 \
    + 0.5 * alpha1 * dot(grad(phi_1 - phi_1_prior), grad(phi_1 - phi_1_prior)) * dx2 \
    + 0.5 * alpha2 * dot(grad(phi_2 - phi_2_prior), grad(phi_2 - phi_2_prior)) * dx2
form_loss_1 = form(loss_element_1)
form_loss_2 = form(loss_element_2)

# vector b_w
b_w_element = u1 * (u - d) * ds_out
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction p
u2 = TestFunction(V2)
j_p = (a2-a3) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 + \
        (a2-a3) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 + \
        (a2-a3) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 + \
        alpha1 * (phi_1 - phi_1_prior) * u2 * dx2 \
        + alpha1 * dot(grad(phi_1 - phi_1_prior), grad(u2)) * dx2
form_J_p = form(j_p, entity_maps=entity_map)
J_p = create_vector(form_J_p)

# vector direction q
j_q = (a2-a3) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 + \
        (a2-a3) * G_phi_1 * delta_deri_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 + \
        (a1-a2) * delta_deri_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 + \
        (a2-a3) * G_phi_1 * delta_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 + \
        (a1-a2) * delta_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 + \
        alpha2 * (phi_2 - phi_2_prior) * u2 * dx2 \
        + alpha2 * dot(grad(phi_2 - phi_2_prior), grad(u2)) * dx2
form_J_q = form(j_p, entity_maps=entity_map)
J_q = create_vector(form_J_q)

# step 1
step1 = np.full(sub_node_num, 0.3)
sub_domain_boundary = locate_entities_boundary(subdomain, tdim - 2, OuterBoundary2)
step1[sub_domain_boundary] = 0.1
step1 = np.diag(step1)
# step 2
step2 = np.full(sub_node_num, 0.8)
step2[sub_domain_boundary] = 0.2
step2 = np.diag(step2)

# phi_2_prior
phi_2_prior.x.array[:] = np.ones(sub_node_num) * tau/2
phi_2_all_time = np.load('2d/phi_2_all_time.npy')

# initial phi_1
phi_1_exact = Function(V2)
phi_1_exact.x.array[:] = np.load(file='2d/phi_1.npy')
phi_0 = np.full(phi_1.x.array.shape, tau/2)
phi_1.x.array[:] = phi_0

# initial phi_2
phi_2.x.array[:] = phi_0

# phi result
phi_1_result = np.zeros((time_total, sub_node_num))
phi_2_result = np.zeros((time_total, sub_node_num))

cm_per_timeframe = []
loss_per_timeframe = []

for timeframe in range(time_total):
    k = 0
    phi_1_prior.x.array[:] = phi_1.x.array
    phi_2_prior.x.array[:] = phi_2.x.array
    # define d's value on the boundary
    d.x.array[:] = d_all_time[timeframe]
    while (k < 1e2):
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
        c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
        u.x.array[:] = u.x.array + c

        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)

        # compute partial derivative of p
        with J_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_p, form_J_p)
        # compute partial derivative of q
        with J_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_q, form_J_q)
        # check if the condition is satisfie
        loss_1 = assemble_scalar(form_loss_1)
        loss_2 = assemble_scalar(form_loss_2)
        loss = loss_1 + loss_2
        if (loss < 1e1 and np.linalg.norm(J_p.array) < 4e0 and np.linalg.norm(J_q.array) < 4e0):
             break
        
        # updata p from partial derivative
        phi_1.x.array[:] = phi_1.x.array - step1@(J_p.array/np.linalg.norm(J_p.array))
        phi_2.x.array[:] = phi_2.x.array - step2@(J_q.array/np.linalg.norm(J_q.array))
        k = k + 1
    print('timeframe:', timeframe)
    print('end at', k, 'iteration')
    print('J_p:', np.linalg.norm(J_p.array))
    print('J_q:', np.linalg.norm(J_q.array))
    print('loss:', loss)
    phi_1_result[timeframe] = phi_1.x.array
    phi_2_result[timeframe] = phi_2.x.array
    cm_per_timeframe.append(compare_CM(subdomain, phi_1_exact, phi_1))
    loss_per_timeframe.append(loss)

# check result
marker_exact = np.zeros(sub_node_num)
marker_exact[phi_1_exact.x.array < 0] = 1
marker_result = np.zeros(sub_node_num)
marker_result[phi_1.x.array < 0] = 1

# phi_1_average = Function(V2)
# phi_1_average.x.array[:] = np.mean(phi_1_result, axis=0)
# print(phi_1_average.x.array)
# print(compare_CM(subdomain, phi_1_exact, phi_1_average))

plt.subplot(1, 2, 1)
plt.plot(cm_per_timeframe)
plt.title('error in center of mass')
plt.xlabel('time')
plt.subplot(1, 2, 2)
plt.plot(loss_per_timeframe)
plt.title('cost functional')
plt.xlabel('time')
plt.show()

grid0 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid0.point_data["marker_exact"] = marker_exact
grid0.set_active_scalars("marker_exact")
grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid1.point_data["marker_reult"] = marker_result
grid1.set_active_scalars("marker_reult")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid2.point_data["phi_1_exact"] = phi_1_exact.x.array
grid2.set_active_scalars("phi_1_exact")
grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid3.point_data["phi_1_result"] = phi_1.x.array
grid3.set_active_scalars("phi_1_result")
grid4 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid4.point_data["phi_2_exact"] = phi_2_all_time[time_total-1]
grid4.set_active_scalars("phi_2_exact")
grid5 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid5.point_data["phi_2_result"] = phi_2.x.array
grid5.set_active_scalars("phi_2_result")

grid = (grid0, grid2, grid4, grid1, grid3, grid5)

plotter = pyvista.Plotter(shape=(2, 3))
for i in range(2):
    for j in range(3):
        plotter.subplot(i, j)
        plotter.add_mesh(grid[i*3+j], show_edges=True)
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