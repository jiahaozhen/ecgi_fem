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
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, eval_function, compute_error_with_v

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
alpha2 = 1e-1

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

# exact phi_1 phi_2 v
phi_1_exact = Function(V2)
phi_1_exact.x.array[:] = np.load(file='2d/data/phi_1.npy')
phi_2_exact_all_time = np.load('2d/data/phi_2_all_time.npy')
v_exact_all_time = np.load('2d/data/v_exact_all_time.npy')

# initial phi_1 phi_2
phi_0 = np.full(phi_1.x.array.shape, tau/2)
phi_1.x.array[:] = phi_0
phi_2.x.array[:] = phi_0

# phi result
phi_1_result = np.zeros((time_total, sub_node_num))
phi_2_result = np.zeros((time_total, sub_node_num))
v_result = np.zeros((time_total, sub_node_num))

cm_phi_1_per_timeframe = []
cm_phi_2_per_timeframe = []
loss_per_timeframe = []

for timeframe in range(time_total):

    # define d's value on the boundary
    d.x.array[:] = d_all_time[timeframe]
    if timeframe < 1:
        phi_1.x.array[:] = phi_1_exact.x.array
        phi_2.x.array[:] = phi_2_exact_all_time[timeframe]
    else:
        phi_2.x.array[:] = phi_2.x.array - tau/2

    # 0-order tikhonov regularization
    # phi_1_est.x.array[:] = np.zeros(phi_1.x.array.shape)
    # phi_2_est.x.array[:] = np.zeros(phi_2.x.array.shape)

    if timeframe == 0:
        phi_1_est.x.array[:] = phi_1.x.array
        phi_2_est.x.array[:] = phi_2.x.array
    elif timeframe == 1:
        phi_1_est.x.array[:] = phi_1_result[timeframe-1]
        phi_2_est.x.array[:] = phi_2_result[timeframe-1]
    else:
        phi_1_est.x.array[:] = phi_1_result[timeframe-1]
        # prior
        # phi_2_est.x.array[:] = phi_2_result[timeframe-1]
        # difference
        phi_2_est.x.array[:] = 2 * phi_2_result[timeframe-1] - phi_2_result[timeframe-2]

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
    loss_in_timeframe = []
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
        if (np.linalg.norm(J_p.array) < 1e0 and np.linalg.norm(J_q.array) < 1e0 and loss < 1e-1):
            break
        
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
        step_search = 0
        while(step_search < 20):
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
            step_search = step_search + 1
        loss_in_timeframe.append(loss_new)
        k = k + 1
    loss_per_timeframe.append(loss)
    phi_1_result[timeframe] = phi_1.x.array.copy()
    phi_2_result[timeframe] = phi_2.x.array.copy()
    v_result[timeframe] = (a1*G_phi_2.x.array + a3*(1-G_phi_2.x.array)) * G_phi_1.x.array + (a2*G_phi_2.x.array + a4*(1-G_phi_2.x.array)) * (1-G_phi_1.x.array)
    cm1, cm2 = compute_error_with_v(v_exact_all_time[timeframe], v_result[timeframe], V2, -90, -60, 10, -20)
    cm_phi_1_per_timeframe.append(cm1)
    cm_phi_2_per_timeframe.append(cm2)
    print('timeframe:', timeframe)
    print('end at', k, 'iteration')
    print('J_p:', np.linalg.norm(J_p.array))
    print('J_q:', np.linalg.norm(J_q.array))
    print('loss:', loss)
    print('error in center of mass (ischemic):', cm1)
    print('error in center of mass (activation):', cm2)
    # plt.plot(loss_in_timeframe)
    # plt.title('loss in each iteration at timeframe ' + str(timeframe))
    # plt.show()

# np.save('2d/data/phi_1_result.npy', phi_1.x.array)
# np.save('2d/data/phi_2_result.npy', phi_2.x.array)
def plot_loss_and_cm():
    plt.subplot(1, 3, 1)
    plt.plot(loss_per_timeframe)
    plt.title('cost functional')
    plt.xlabel('time')
    plt.subplot(1, 3, 2)
    plt.plot(cm_phi_1_per_timeframe)
    plt.title('error in center of mass (ischemic)')
    plt.xlabel('time')
    plt.subplot(1, 3, 3)
    plt.plot(cm_phi_2_per_timeframe)
    plt.title('error in center of mass (activation)')
    plt.xlabel('time')
    plt.show()

def plot_with_time(value, title):
    v_function = Function(V2)
    plotter = pyvista.Plotter(shape=(2, 5))
    for i in range(2):
        for j in range(5):
            plotter.subplot(i, j)
            grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
            v_function.x.array[:] = value[i*20 + j*5]
            grid.point_data[title] = eval_function(v_function, subdomain.geometry.x)
            grid.set_active_scalars(title)
            plotter.add_mesh(grid, show_edges=True)
            plotter.add_text(f"Time: {i*20 + j*5:.1f} ms", position='lower_right', font_size=9)
            plotter.view_xy()
            plotter.add_title(title, font_size=9)
    plotter.show()

p1 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_result < 0, 1, 0), 'ischemic'))
p2 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_result < 0, 1, 0), 'activation_result'))
p3 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_exact_all_time < 0, 1, 0), 'activation_exact'))
p4 = multiprocessing.Process(target=plot_with_time, args=(v_exact_all_time, 'v'))
p5 = multiprocessing.Process(target=plot_loss_and_cm)
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()