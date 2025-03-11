import sys
import time

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure, derivative, sqrt, inner
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, compute_error_with_v, eval_function, compute_phi_with_v_timebased

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
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local
sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))

# sigma_i : intra-cellular conductivity tensor in Heart
# sigma_e : extra-cellular conductivity tensor in Heart
# sigma_t : conductivity tensor in Torso
# M  : sigma_i + sigma_e in Heart 
#      sigma_t in Torso
# S/m
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
def rho3(x):
    tensor = np.eye(tdim) * sigma_t / 5
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho4(x):
    tensor = np.eye(tdim) * sigma_t * 3
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])

M = Function(V3)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
if cell_markers.find(3).any():
    M.interpolate(rho3, cell_markers.find(3))
if cell_markers.find(4).any():
    M.interpolate(rho4, cell_markers.find(4))
Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim)*sigma_i))

# paramter
a1 = -90 # no active no ischemia
a2 = -60 # no active ischemia
a3 = 10  # active no ischemia
a4 = -20 # active ischemia
tau = 1
alpha1 = 1e0
alpha2 = 1e0

# phi phi_est G_phi delta_phi delta_deri_phi
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

# function u w d
u = Function(V1)
w = Function(V1)
d = Function(V1)
v = Function(V2)
# define d's value on the boundary
d_all_time = np.load(d_data_file)
time_total = np.shape(d_all_time)[0]

# matrix A_u
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
dx1 = Measure("dx", domain = domain)
a_element = dot(grad(v1), dot(M, grad(u1))) * dx1
bilinear_form_a = form(a_element)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()

solver = PETSc.KSP().create()
solver.setOperators(A_u)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# vector b_u
dx2 = Measure("dx",domain = subdomain_ventricle)
b_u_element = -dot(grad(v1), dot(Mi, grad(v))) * dx2
# b_u_element = -(a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 \
#         - (a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2 \
#         - (a3 - a4) * delta_phi_1 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 \
#         - (a2 - a4) * delta_phi_2 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2
entity_map = {domain._cpp_object: ventricle_to_torso}
linear_form_b_u = form(b_u_element, entity_maps = entity_map)
b_u = create_vector(linear_form_b_u)

# scalar c
ds = Measure('ds', domain = domain)
c1_element = (d - u) * ds
c2_element = 1 * ds
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element_1 = 0.5 * (u - d) ** 2 * ds
loss_element_2 = 0.5 * alpha1 * (phi_1 - phi_1_est) ** 2 * dx2 + 0.5 * alpha2 * (phi_2 - phi_2_est) ** 2 * dx2 \
    # + 0.5 * alpha1 * dot(grad(phi_1 - phi_1_prior), grad(phi_1 - phi_1_prior)) * dx2 \
    # + 0.5 * alpha2 * dot(grad(phi_2 - phi_2_prior), grad(phi_2 - phi_2_prior)) * dx2
reg_element = alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2 + \
                alpha2 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2

form_loss_1 = form(loss_element_1)
form_loss_2 = form(loss_element_2)
form_reg = form(reg_element)

# vector b_w
b_w_element = u1 * (u - d) * ds
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction
u2 = TestFunction(V2)
# F1 = alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1))) * dx2
j_p = -(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a1 - a2) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a1 - a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        - (a3 - a4) * delta_deri_phi_1 * (1 - G_phi_2) * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a3 - a4) * delta_phi_1 * (1 - G_phi_2) * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        # + alpha1 * (phi_1 - phi_1_est) * u2 * dx2
form_J_p = form(j_p, entity_maps = entity_map)
form_Reg_p = form(derivative(alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2, phi_1, u2), entity_maps = entity_map)
J_p = create_vector(form_J_p)
Reg_p = create_vector(form_Reg_p)

# F2 = alpha2 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2))) * dx2
j_q = -(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 \
        - (a1 - a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a1 - a3) * delta_phi_2 * G_phi_1 * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        - (a2 - a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 \
        - (a2 - a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2 \
        # + alpha2 * (phi_2 - phi_2_est) * u2 * dx2
form_J_q = form(j_q, entity_maps = entity_map)
form_Reg_q = form(derivative(alpha2 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2, phi_2, u2), entity_maps = entity_map)
J_q = create_vector(form_J_q)
Reg_q = create_vector(form_Reg_q)

# exact v
v_exact = np.load(v_exact_data_file)

#initial phi_1 phi_2
phi_0 = np.full(phi_1.x.array.shape, tau/2)

# phi result
phi_1_result = np.zeros((time_total, sub_node_num))
phi_2_result = np.zeros((time_total, sub_node_num))
v_result = np.zeros((time_total, sub_node_num))

cm_phi_1_per_timeframe = []
cm_phi_2_per_timeframe = []
loss_per_timeframe = []

for timeframe in range(time_total):

    start_time = time.time()

    if timeframe % 10 == 0:
        phi_1.x.array[:] =  phi_0
        if timeframe < 50:
            phi_2.x.array[:] =  phi_0
        else:
            phi_2.x.array[:] = -phi_0

    d.x.array[:] = d_all_time[timeframe]

    # 0-order tikhonov regularization
    # phi_1_est.x.array[:] = np.zeros(phi_1.x.array.shape)
    # phi_2_est.x.array[:] = np.zeros(phi_2.x.array.shape)

    # if timeframe == 0:
    #     phi_1_est.x.array[:] = phi_1.x.array
    #     phi_2_est.x.array[:] = phi_2.x.array
    # elif timeframe == 1:
    #     phi_1_est.x.array[:] = phi_1_result[timeframe-1]
    #     phi_2_est.x.array[:] = phi_2_result[timeframe-1]
    # else:
    #     phi_1_est.x.array[:] = phi_1_result[timeframe-1]
    #     # prior
    #     phi_2_est.x.array[:] = phi_2_result[timeframe-1]
    #     # difference
    #     phi_2_est.x.array[:] = 2 * phi_2_result[timeframe-1] - phi_2_result[timeframe-2]

    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
    delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
    delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
    v.x.array[:] = (a1*G_phi_2.x.array + a3*(1-G_phi_2.x.array)) * G_phi_1.x.array + (a2*G_phi_2.x.array + a4*(1-G_phi_2.x.array)) * (1-G_phi_1.x.array)

    # get u from p, q
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    k = 0
    loss_in_timeframe = []
    cc = []
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
        # J_p  = J_p + assemble_vector(form_Reg_p)
        with Reg_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(Reg_p, form_Reg_p)
        J_p = J_p + Reg_p
        with J_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_q, form_J_q)
        # print('the norm of J_p:', np.linalg.norm(J_p.array))
        # print('the norm of J_q:', np.linalg.norm(J_q.array))
        # J_q = J_q + assemble_vector(form_Reg_q)
        with Reg_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(Reg_q, form_Reg_q)
        J_q = J_q + Reg_q
        # cost function
        loss = assemble_scalar(form_loss_1) \
            + assemble_scalar(form_reg) \
            # + assemble_scalar(form_loss_2)

        # check if the condition is satisfie
        if (np.linalg.norm(J_p.array) < 1e-1 and np.linalg.norm(J_q.array) < 1e-1):
             break
        
        # line search
        phi_1_v = phi_1.x.array[:].copy()
        phi_2_v = phi_2.x.array[:].copy()
        dir_p = J_p.array.copy() / max(J_p.array) * tau * 2
        dir_q = J_q.array.copy() / max(J_q.array) * tau * 2
        # functionspace2mesh = fspace2mesh(V2)
        # mesh2functionspace = np.argsort(functionspace2mesh)
        # J_p_array[mesh2functionspace[sub_domain_boundary]] = J_p_array[mesh2functionspace[sub_domain_boundary]]
        # J_q_array[mesh2functionspace[sub_domain_boundary]] = J_q_array[mesh2functionspace[sub_domain_boundary]]
        alpha = 1
        gamma = 0.6
        c = 0.1
        step_search = 0
        while(True):
            # adjust p q
            # different step maybe helpful
            phi_1.x.array[:] = phi_1_v - alpha * dir_p
            phi_2.x.array[:] = phi_2_v - alpha * dir_q
            # get u from p, q
            G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
            delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
            v.x.array[:] = (a1*G_phi_2.x.array + a3*(1-G_phi_2.x.array)) * G_phi_1.x.array + (a2*G_phi_2.x.array + a4*(1-G_phi_2.x.array)) * (1-G_phi_1.x.array)
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment
            # compute loss
            loss_new = assemble_scalar(form_loss_1) \
                 + assemble_scalar(form_reg) \
                #  + assemble_scalar(form_loss_2)
            loss_cmp = loss_new - (loss - c * alpha * np.concatenate((J_p.array, J_q.array)).dot(np.concatenate((dir_p, dir_q))))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 20 or loss_cmp < 0):
                loss = loss_new
                with J_p.localForm() as loc_J:
                    loc_J.set(0)
                assemble_vector(J_p, form_J_p)
                # J_p  = J_p + assemble_vector(form_Reg_p)
                with Reg_p.localForm() as loc_J:
                    loc_J.set(0)
                assemble_vector(Reg_p, form_Reg_p)
                J_p = J_p + Reg_p
                with J_q.localForm() as loc_J:
                    loc_J.set(0)
                assemble_vector(J_q, form_J_q)
                # J_q = J_q + assemble_vector(form_Reg_q)
                with Reg_q.localForm() as loc_J:
                    loc_J.set(0)
                assemble_vector(Reg_q, form_Reg_q)
                J_q = J_q + Reg_q
                break
        loss_in_timeframe.append(loss)
        k = k + 1

    loss_per_timeframe.append(loss)
    phi_1_result[timeframe] = phi_1.x.array.copy()
    phi_2_result[timeframe] = phi_2.x.array.copy()
    v_result[timeframe] = (a1*G_phi_2.x.array + a3*(1-G_phi_2.x.array)) * G_phi_1.x.array + (a2*G_phi_2.x.array + a4*(1-G_phi_2.x.array)) * (1-G_phi_1.x.array)
    cm1, cm2 = compute_error_with_v(v_exact[timeframe], v_result[timeframe], V2, -90, -60, 10, -20)
    cm_phi_1_per_timeframe.append(cm1)
    cm_phi_2_per_timeframe.append(cm2)
    end_time = time.time()
    print('timeframe:', timeframe)
    print('end at', k, 'iteration')
    print('J_p:', np.linalg.norm(J_p.array))
    print('J_q:', np.linalg.norm(J_q.array))
    print('loss:', loss)
    print('error in center of mass (ischemic):', cm1)
    print('error in center of mass (activation):', cm2)
    print(f"cost {end_time - start_time} seconds")
    cc.append(np.corrcoef(v_exact[timeframe], v_result[timeframe])[0, 1])
    # plt.plot(loss_in_timeframe)
    # plt.title('loss in each iteration at timeframe ' + str(timeframe))
    # plt.show()

if gdim == 2:
    np.save('2d/data/phi_1_result.npy', phi_1_result)
    np.save('2d/data/phi_2_result.npy', phi_2_result)
    np.save('2d/data/v_result.npy', v_result)
else:
    np.save('3d/data/phi_1_result.npy', phi_1_result)
    np.save('3d/data/phi_2_result.npy', phi_2_result)
    np.save('3d/data/v_result.npy', v_result)

def plot_loss():
    # plt.subplot(1, 3, 1)
    plt.subplot(1, 2, 1)
    plt.plot(loss_per_timeframe)
    plt.title('cost functional')
    plt.xlabel('time')
    plt.subplot(1, 2, 2)
    plt.plot(cc)
    plt.title('correlation coefficient')
    plt.xlabel('time')
    plt.show()
    # plt.subplot(1, 3, 2)
    # plt.plot(cm_phi_1_per_timeframe)
    # plt.title('error in center of mass (ischemic)')
    # plt.xlabel('time')
    # plt.subplot(1, 3, 3)
    # plt.plot(cm_phi_2_per_timeframe)
    # plt.title('error in center of mass (activation)')
    # plt.xlabel('time')
    # plt.show()

def plot_with_time(value, title):
    v_function = Function(V2)
    plotter = pyvista.Plotter(shape=(3, 7))
    for i in range(3):
        for j in range(7):
            plotter.subplot(i, j)
            grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
            v_function.x.array[:] = value[i*70 + j*10]
            grid.point_data[title] = eval_function(v_function, subdomain_ventricle.geometry.x)
            grid.set_active_scalars(title)
            plotter.add_mesh(grid, show_edges=True)
            plotter.add_text(f"Time: {(i*70 + j*10)/5.0:.1f} ms", position='lower_right', font_size=9)
            plotter.view_xy()
            plotter.add_title(title, font_size=9)
    plotter.show()

p1 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_result < 0, 1, 0), 'ischemic'))
p2 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_result < 0, 1, 0), 'activation'))
p3 = multiprocessing.Process(target=plot_with_time, args=(v_result, 'v_result'))
p4 = multiprocessing.Process(target=plot_with_time, args=(v_exact, 'v_exact'))
p5 = multiprocessing.Process(target=plot_loss)
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