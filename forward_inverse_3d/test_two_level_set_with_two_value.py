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
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, eval_function

def ischemia_inversion(mesh_file, d_data, v_exact, tau, alpha1, alpha2, alpha3,  
                       gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                       a1=-90, a2=-60, a3=10, a4=-20,
                       plot_flag=False, print_message=False):

    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    # function space
    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
    V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))

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
    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))

    # phi phi_est G_phi delta_phi delta_deri_phi
    phi_1 = Function(V2)
    phi_2 = Function(V2)
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
    dx2 = Measure("dx", domain=subdomain_ventricle)
    b_u_element = - (a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(v1), dot(Mi, grad(phi_1))) * dx2 \
                  - (a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(v1), dot(Mi, grad(phi_2))) * dx2 \
                  - (a3 - a4) * delta_phi_1 * dot(grad(v1), dot(Mi, grad(phi_1))) * dx2 \
                  - (a2 - a4) * delta_phi_2 * dot(grad(v1), dot(Mi, grad(phi_2))) * dx2
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b_u = form(b_u_element, entity_maps=entity_map)
    b_u = create_vector(linear_form_b_u)

    # scalar c
    ds = Measure('ds', domain=domain)
    c1_element = (d - u) * ds
    c2_element = 1 * ds
    form_c1 = form(c1_element)
    form_c2 = form(c2_element)

    # scalar loss
    loss_element = 0.5 * (u - d) ** 2 * ds
    reg_element = alpha3 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2 + \
                  alpha3 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2

    form_loss = form(loss_element)
    form_reg = form(reg_element)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    residual_p = (-(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a1 - a2) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a1 - a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2
                  - (a3 - a4) * delta_deri_phi_1 * (1 - G_phi_2) * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a3 - a4) * delta_phi_1 * (1 - G_phi_2) * dot(grad(w), dot(Mi, grad(u2))) * dx2)
    reg_p = derivative(alpha3 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2, phi_1, u2)
    form_Residual_p = form(residual_p, entity_maps=entity_map)
    form_Reg_p = form(reg_p, entity_maps=entity_map)
    J_p = create_vector(form_Residual_p)
    Reg_p = create_vector(form_Reg_p)

    residual_q = (-(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a1 - a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a1 - a3) * delta_phi_2 * G_phi_1 * dot(grad(w), dot(Mi, grad(u2))) * dx2 
                  - (a2 - a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a2 - a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2)
    reg_q = derivative(alpha3 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2, phi_2, u2)
    form_Residual_q = form(residual_q, entity_maps=entity_map)
    form_Reg_q = form(reg_q, entity_maps=entity_map)
    J_q = create_vector(form_Residual_q)
    Reg_q = create_vector(form_Reg_q)

    # initial phi_1, phi_2
    phi_1.x.array[:] = np.full(phi_1.x.array.shape, tau/2)
    phi_2.x.array[:] = np.full(phi_1.x.array.shape, tau/2)

    d.x.array[:] = d_data

    # prepare to compute u from phi_1 phi_2
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)

    # get u from v
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    k = 0
    loss_in_timeframe = []
    while (True):
        # prepare to compute partial derivative of p, q
        delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
        delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)

        # cost function
        loss = assemble_scalar(form_loss)
        loss_in_timeframe.append(loss)
        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)
        # compute partial derivative of p, q from w
        with J_p.localForm() as loc_jp:
            loc_jp.set(0)
        assemble_vector(J_p, form_Residual_p)
        # with Reg_p.localForm() as loc_rp:
        #     loc_rp.set(0)
        # assemble_vector(Reg_p, form_Reg_p)
        # J_p.axpy(1.0, Reg_p)
        with J_q.localForm() as loc_jq:
            loc_jq.set(0)
        assemble_vector(J_q, form_Residual_q)
        # with Reg_q.localForm() as loc_rq:
        #     loc_rq.set(0)
        # assemble_vector(Reg_q, form_Reg_q)
        # J_q.axpy(1.0, Reg_q)
        print('iteration:', k)
        print('loss_residual:', assemble_scalar(form_loss))
        print('J_p', np.linalg.norm(J_p.array))
        print('J_q', np.linalg.norm(J_q.array))
        # check if the condition is satisfied
        if (k > 2e2 or np.linalg.norm(J_p.array) < 1e-1 and np.linalg.norm(J_q.array) < 1e-1):
            break
        k = k + 1

        # line search
        phi_1_v = phi_1.x.array[:].copy()
        phi_2_v = phi_2.x.array[:].copy()
        dir_p = -J_p.array.copy()
        dir_q = -J_q.array.copy()
            # sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, 
            #                                                lambda x: np.full(x.shape[1], True, dtype=bool))
            # functionspace2mesh = fspace2mesh(V2)
            # mesh2functionspace = np.argsort(functionspace2mesh)
            # J_p_array[mesh2functionspace[sub_domain_boundary]] = J_p_array[mesh2functionspace[sub_domain_boundary]]
            # J_q_array[mesh2functionspace[sub_domain_boundary]] = J_q_array[mesh2functionspace[sub_domain_boundary]]
        alpha = 1
        gamma = 0.8
        c = 0.1
        step_search = 0
        while(True):
            # adjust p q
            phi_1.x.array[:] = phi_1_v + alpha * dir_p
            phi_2.x.array[:] = phi_2_v + alpha * dir_q
            G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            # get u from p, q
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment

            # compute loss
            loss_new = assemble_scalar(form_loss)
            loss_cmp = loss_new - (loss + c * alpha * np.concatenate((J_p.array, J_q.array))
                                    .dot(np.concatenate((dir_p, dir_q))))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 100 or loss_cmp < 0):
                break
        
    v_result = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
    cc = np.corrcoef(v_exact, v_result)[0, 1]
    # cc_per_timeframe.append(cc)
    if print_message == True:
        print('end at', k, 'iteration')
        print('J_p:', np.linalg.norm(J_p.array))
        print('J_q:', np.linalg.norm(J_q.array))
        print('loss_residual:', assemble_scalar(form_loss))
        print('correlation coefficient:', cc)
        # plt.plot(loss_in_timeframe)
        # plt.title('loss in each iteration at timeframe 0')
        # plt.show()

    if plot_flag == False:
        return phi_1.x.array, phi_2.x.array, v_result

    def plot_with_time(value, title):
        v_function = Function(V2)
        plotter = pyvista.Plotter()
        grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
        v_function.x.array[:] = value
        grid.point_data[title] = eval_function(v_function, subdomain_ventricle.geometry.x)
        grid.set_active_scalars(title)
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.add_title(title, font_size=9)
        plotter.show()

    p1 = multiprocessing.Process(target=plot_with_time, args=(phi_1.x.array, 'ischemia'))
    p2 = multiprocessing.Process(target=plot_with_time, args=(phi_2.x.array, 'activation'))
    p3 = multiprocessing.Process(target=plot_with_time, args=(v_result, 'v_result'))
    p4 = multiprocessing.Process(target=plot_with_time, args=(v_exact, 'v_exact'))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    return phi_1.x.array, phi_2.x.array, v_result

if __name__ == '__main__':
    gdim = 3
    mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    v_exact_data_file = '3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy'
    d_data_file = '3d/data/u_data_reaction_diffusion_ischemia_data_argument.npy'
    v_exact = np.load(v_exact_data_file)[800]
    d_data = np.load(d_data_file)[800]
    phi_1, phi_2, v_result = ischemia_inversion(mesh_file=mesh_file, d_data=d_data, v_exact=v_exact, gdim=3, 
                                                tau=10, alpha1=1e0, alpha2=1e0, alpha3=5e-20, 
                                                plot_flag=True, print_message=True)