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
from main_ischemia import resting_ischemia_inversion
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, eval_function, compute_error_phi, find_vertex_with_neighbour_less_than_0

def ischemia_inversion(mesh_file, d_data, v_exact, tau, alpha1, alpha2, alpha3, alpha4,
                       phi_1_exact = np.load('3d/data/phi_1_exact_reaction_diffusion.npy'), 
                       phi_2_exact = np.load('3d/data/phi_2_exact_reaction_diffusion.npy'), 
                       gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                       a1=-90, a2=-80, a3=10, a4=0,
                       plot_flag=False, print_message=False):

    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    sub_node_num = subdomain_ventricle.topology.index_map(0).size_local

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
    phi_1_est = Function(V2)
    phi_2_est = Function(V2)
    G_phi_1 = Function(V2)
    G_phi_2 = Function(V2)
    delta_phi_1 = Function(V2)
    delta_phi_2 = Function(V2)
    delta_deri_phi_1 = Function(V2)
    delta_deri_phi_2 = Function(V2)
    phi_2_mono = Function(V2)
    phi_2_I = Function(V2)

    # function u w d
    u = Function(V1)
    w = Function(V1)
    d = Function(V1)
    v = Function(V2)
    # define d's value on the boundary
    time_total = np.shape(d_data)[0]

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
    dx2 = Measure("dx",domain=subdomain_ventricle)
    b_u_element = -dot(grad(v1), dot(Mi, grad(v))) * dx2
    # b_u_element = -(a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(v1), dot(Mi, grad(phi_1))) * dx2 \
    #         - (a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(v1), dot(Mi, grad(phi_2))) * dx2 \
    #         - (a3 - a4) * delta_phi_1 * dot(grad(v1), dot(Mi, grad(phi_1))) * dx2 \
    #         - (a2 - a4) * delta_phi_2 * dot(grad(v1), dot(Mi, grad(phi_2))) * dx2
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
    reg_element_1 = 0.5 * alpha1 * (phi_1 - phi_1_est) ** 2 * dx2
    reg_ekement_2 = 0.5 * alpha2 * phi_2_mono ** 2 * dx2
    reg_element_3 = alpha3 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2
    reg_element_4 = alpha4 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2

    form_loss = form(loss_element)

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
    reg_p_1 = alpha1 * (phi_1 - phi_1_est) * u2 * dx2
    reg_p_2 = derivative(alpha3 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2, phi_1, u2)
    form_Residual_p = form(residual_p, entity_maps=entity_map)
    form_Reg_p = form(reg_p_2, entity_maps=entity_map)
    J_p = create_vector(form_Residual_p)
    Reg_p = create_vector(form_Reg_p)

    residual_q = (-(a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a1 - a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a1 - a3) * delta_phi_2 * G_phi_1 * dot(grad(w), dot(Mi, grad(u2))) * dx2 
                  - (a2 - a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a2 - a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2)
    reg_q_1 = alpha2 * phi_2_mono * phi_2_I * u2 * dx2
    reg_q_2 = derivative(alpha4 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2, phi_2, u2)
    form_Residual_q = form(residual_q, entity_maps=entity_map)
    form_Reg_q = form(reg_q_1 + reg_q_2, entity_maps=entity_map)
    J_q = create_vector(form_Residual_q)
    Reg_q = create_vector(form_Reg_q)

    # initial phi_1
    print('start copmuting phi_1')
    phi_1_init = resting_ischemia_inversion(mesh_file, d_data=d_data[0], gdim=gdim,
                                            ischemia_potential=a2, normal_potential=a1, tau=tau)
    #initial phi_2
    phi_1.x.array[:] = phi_1_init
    # phi_2.x.array[:] = np.full(phi_2.x.array.shape, tau/2)
    phi_2.x.array[:] = np.where(phi_2_exact[0] < 0, -tau/2, tau/2)

    # phi result
    phi_1_result = np.zeros((time_total, sub_node_num))
    phi_2_result = np.zeros((time_total, sub_node_num))
    v_result_1 = np.zeros((time_total, sub_node_num))
    v_result_2 = np.zeros((time_total, sub_node_num))

    cm_phi_2_per_timeframe = []
    loss_1_per_timeframe = []
    cc_1_per_timeframe = []
    cc_2_per_timeframe = []

    # fix phi_1 for phi_2
    print('start computing phi_2 with phi_1 fixed')
    form_reg = form(reg_ekement_2 + reg_element_4, entity_maps=entity_map)
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
    for timeframe in range(time_total):

        start_time = time.time()

        d.x.array[:] = d_data[timeframe]
        if timeframe == 0:
            phi_2_est.x.array[:] = phi_2.x.array
        else:
            phi_2_est.x.array[:] = phi_2_result[timeframe-1]

        # prepare to compute u from phi_1 phi_2
        G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
        v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                        (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))

        # get u from v
        with b_u.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b_u, linear_form_b_u)
        solver.solve(b_u, u.vector)
        # adjust u
        adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
        u.x.array[:] = u.x.array + adjustment
        delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
        phi_2_mono.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0, 
                                             phi_2.x.array - phi_2_est.x.array, 0)

        k = 0
        iter_total = 1e2
        loss_in_timeframe = []
        while (True):
            # cost function
            loss = assemble_scalar(form_loss) + assemble_scalar(form_reg)
            loss_in_timeframe.append(loss)

            # prepare to compute partial derivative of q
            delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
            phi_2_I.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0, 1, 0)
            # get w from u
            with b_w.localForm() as loc_w:
                loc_w.set(0)
            assemble_vector(b_w, linear_form_b_w)
            solver.solve(b_w, w.vector)
            # compute partial derivative of q from w
            with J_q.localForm() as loc_jq:
                loc_jq.set(0)
            assemble_vector(J_q, form_Residual_q)
            with Reg_q.localForm() as loc_rq:
                loc_rq.set(0)
            assemble_vector(Reg_q, form_Reg_q)
            J_q.axpy(1.0, Reg_q)
        
            # check if the condition is satisfied
            if (k > iter_total or np.linalg.norm(J_q.array) < 1e-1):
                phi_2.x.array[:] = np.where(phi_2.x.array < 0, phi_2.x.array - tau / 2 , phi_2.x.array)
                neighbour_idx, neighbour_map = find_vertex_with_neighbour_less_than_0(subdomain_ventricle, phi_2)
                neighbour_weight = [neighbour_map[i] for i in neighbour_idx]
                phi_2.x.array[neighbour_idx] = np.where(phi_2.x.array[neighbour_idx] >= 0,
                                                        phi_2.x.array[neighbour_idx] - 2 * np.array(neighbour_weight) * tau / time_total, 
                                                        phi_2.x.array[neighbour_idx])
                G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
                delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
                delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
                phi_2_I.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0, 1, 0)
                phi_2_mono.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0,
                                                    phi_2.x.array - phi_2_est.x.array, 0)
                v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                                (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
                with b_u.localForm() as loc_b:
                    loc_b.set(0)
                assemble_vector(b_u, linear_form_b_u)
                solver.solve(b_u, u.vector)
                # adjust u
                adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                u.x.array[:] = u.x.array + adjustment
                loss = assemble_scalar(form_loss) + assemble_scalar(form_reg)
                # get w from u
                with b_w.localForm() as loc_w:
                    loc_w.set(0)
                assemble_vector(b_w, linear_form_b_w)
                solver.solve(b_w, w.vector)
                # compute partial derivative of q from w
                with J_q.localForm() as loc_jq:
                    loc_jq.set(0)
                assemble_vector(J_q, form_Residual_q)
                with Reg_q.localForm() as loc_rq:
                    loc_rq.set(0)
                assemble_vector(Reg_q, form_Reg_q)
                J_q.axpy(1.0, Reg_q)
                break
            k = k + 1

            # line search
            phi_2_v = phi_2.x.array[:].copy()
            dir_q = -J_q.array.copy()
            alpha = 1
            gamma = 0.8
            c = 1e-1
            step_search = 0
            while(True):
                # adjust q
                phi_2.x.array[:] = phi_2_v + alpha * dir_q
                # get u from q
                G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
                delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
                phi_2_mono.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0,
                                                    phi_2.x.array - phi_2_est.x.array, 0)
                v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                                (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
                with b_u.localForm() as loc_b:
                    loc_b.set(0)
                assemble_vector(b_u, linear_form_b_u)
                solver.solve(b_u, u.vector)
                # adjust u
                adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                u.x.array[:] = u.x.array + adjustment

                # compute loss
                loss_new = assemble_scalar(form_loss) + assemble_scalar(form_reg)
                loss_diff = c * alpha * J_q.array.dot(dir_q)
                loss_target = loss + loss_diff
                loss_cmp = loss_new - loss_target
                alpha = gamma * alpha
                step_search = step_search + 1
                if (step_search > 1e2 or loss_cmp < 0):
                    # neighbour_idx = find_vertex_with_neighbour_less_than_0(subdomain_ventricle, phi_2)
                    # phi_2.x.array[neighbour_idx] = np.where(phi_2.x.array[neighbour_idx] >= 0, 
                    #                                         phi_2.x.array[neighbour_idx] - tau / iter_total, 
                    #                                         phi_2.x.array[neighbour_idx])
                    # G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
                    # delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
                    # phi_2_mono.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0,
                    #                                     phi_2.x.array - phi_2_est.x.array, 0)
                    # v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                    #                 (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
                    # with b_u.localForm() as loc_b:
                    #     loc_b.set(0)
                    # assemble_vector(b_u, linear_form_b_u)
                    # solver.solve(b_u, u.vector)
                    # # adjust u
                    # adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                    # u.x.array[:] = u.x.array + adjustment
                    break

        end_time = time.time()
        loss_1_per_timeframe.append(loss)
        phi_2_result[timeframe] = phi_2.x.array.copy()
        v_result_1[timeframe] = v.x.array.copy()
        cm2 = compute_error_phi(phi_2.x.array, phi_2_exact[timeframe], V2)
        cm_phi_2_per_timeframe.append(cm2)
        cc = np.corrcoef(v_exact[timeframe], v_result_1[timeframe])[0, 1]
        cc_1_per_timeframe.append(cc)
        if print_message == True:
            print('timeframe:', timeframe)
            print('end at', k, 'iteration')
            print('J_q:', np.linalg.norm(J_q.array))
            print('loss_residual:', assemble_scalar(form_loss))
            print('loss_reg_mono:', assemble_scalar(form(reg_ekement_2, entity_maps=entity_map)))
            print('loss_reg_grad:', assemble_scalar(form(reg_element_4, entity_maps=entity_map)))
            print('error in center of mass (activation):', cm2)
            print('correlation coefficient of v:', cc)
            print(f"cost {end_time - start_time} seconds")

    if gdim == 2:
        np.save('2d/data/phi_2_result.npy', phi_2_result)
    else:
        np.save('3d/data/phi_2_result.npy', phi_2_result)
    
    # phi_2_result = np.load('3d/data/phi_2_result.npy')

    # fix phi_2 for phi_1
    print('start adjust phi_1 with phi_2 fixed')
    form_reg = form(reg_element_3, entity_maps=entity_map)

    def compute_u_from_phi_1(phi_1: Function):
        G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
        u_array = np.zeros((time_total, domain.topology.index_map(0).size_local))
        for timeframe in range(time_total):
            d.x.array[:] = d_data[timeframe]
            phi_2.x.array[:] = phi_2_result[timeframe]
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)    
            v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array +
                            (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
            # get u from v
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment

            u_array[timeframe] = u.x.array.copy()
        return u_array

    def compute_Jp_from_phi_1(phi_1: Function, u_array: np.ndarray):
        if u_array is None:
            u_array = compute_u_from_phi_1(phi_1)
        J_p_array = np.full_like(phi_1.x.array, 0)
        G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
        for timeframe in range(time_total):
            d.x.array[:] = d_data[timeframe]
            u.x.array[:] = u_array[timeframe]
            phi_2.x.array[:] = phi_2_result[timeframe]
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
            # get w from u
            with b_w.localForm() as loc_w:
                loc_w.set(0)
            assemble_vector(b_w, linear_form_b_w)
            solver.solve(b_w, w.vector)
            # compute partial derivative of p from w
            with J_p.localForm() as loc_jp:
                loc_jp.set(0)
            assemble_vector(J_p, form_Residual_p)
            with Reg_p.localForm() as loc_rp:
                loc_rp.set(0)
            assemble_vector(Reg_p, form_Reg_p)
            J_p_array  = J_p_array + J_p.array.copy() + Reg_p.array.copy()
        return J_p_array
        
    def compute_loss_from_phi_1(phi_1: Function, u_array: np.ndarray):
        if u_array is None:
            u_array = compute_u_from_phi_1(phi_1)
        loss_residual = 0
        loss_reg = 0
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        for timeframe in range(time_total):    
            d.x.array[:] = d_data[timeframe]
            u.x.array[:] = u_array[timeframe]
            loss_residual = loss_residual + assemble_scalar(form_loss)
            loss_reg = loss_reg + assemble_scalar(form_reg)
        return loss_residual, loss_reg

    loss_per_iter = []
    cm_phi_1_per_iter = []
    k = 0
    u_array = compute_u_from_phi_1(phi_1)
    start_time = time.time()
    while (True):
        loss_residual, loss_reg = compute_loss_from_phi_1(phi_1, u_array)
        loss = loss_residual + loss_reg

        loss_per_iter.append(loss)
        cm1 = compute_error_phi(phi_1.x.array, phi_1_exact[0], V2)
        cm_phi_1_per_iter.append(cm1)
        J_p_array =  compute_Jp_from_phi_1(phi_1, u_array)
        end_time = time.time()
        print('iteration:', k)
        print('loss_residual:', loss_residual)
        print('loss_reg:', loss_reg)
        print('J_p:', np.linalg.norm(J_p_array))
        print('center of mass error:', cm1)
        print('cost', end_time - start_time, 'seconds')
        if (k > 1e2 or np.linalg.norm(J_p_array) < 1e-1):
            break
        k = k + 1
        start_time = time.time()

        phi_1_v = phi_1.x.array[:].copy()
        dir_p = -J_p_array.copy()
        alpha = 1
        gamma = 0.6
        c = 1e-3
        step_search = 0
        while(True):
            # adjust p
            phi_1.x.array[:] = phi_1_v + alpha * dir_p
            # compute u
            u_array = compute_u_from_phi_1(phi_1)
            # compute loss
            loss_residual_new, loss_reg_new = compute_loss_from_phi_1(phi_1, u_array)
            loss_new = loss_residual_new + loss_reg_new
            loss_cmp = loss_new - (loss + c * alpha * J_p_array.dot(dir_p))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 1e2 or loss_cmp < 0):
                break
    
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau/10)
    for timeframe in range(time_total):
        phi_1_result[timeframe] = phi_1.x.array.copy()
        phi_2.x.array[:] = phi_2_result[timeframe]
        
        G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau/10)
        v_result_2[timeframe] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                                 (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))

    if gdim == 2:
        np.save('2d/data/phi_1_result.npy', phi_1_result)
        np.save('2d/data/v_result.npy', v_result_2)
    else:
        np.save('3d/data/phi_1_result.npy', phi_1_result)
        np.save('3d/data/v_result.npy', v_result_2)

    if plot_flag == False:
        return phi_1_result, phi_2_result, v_result_2

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
                plotter.add_text(f"Time: {i*14 + j*2:.1f} ms", position='lower_right', font_size=9)
                plotter.view_xy()
                plotter.add_title(title, font_size=9)
        plotter.show()

    p1 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_result < 0, 1, 0), 'ischemia'))
    p2 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_result < 0, 1, 0), 'activation'))
    p3 = multiprocessing.Process(target=plot_with_time, args=(v_result_2, 'v_result'))
    p4 = multiprocessing.Process(target=plot_with_time, args=(v_exact, 'v_exact'))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    return phi_1_result, phi_2_result, v_result_2

if __name__ == '__main__':
    gdim = 3
    if gdim == 2:
        mesh_file = '2d/data/heart_torso.msh'
        v_exact_data_file = '2d/data/v_data_reaction_diffusion.npy'
        d_data_file = '2d/data/u_data_reaction_diffusion.npy'
    else:
        mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
        v_exact_data_file = '3d/data/v_data_reaction_diffusion.npy'
        d_data_file = '3d/data/u_data_reaction_diffusion.npy'
    v_exact = np.load(v_exact_data_file)
    d_data = np.load(d_data_file)
    phi_1, phi_2, v_result = ischemia_inversion(mesh_file=mesh_file, d_data=d_data, v_exact=v_exact, gdim=3, 
                                                tau=1, alpha1=1e0, alpha2=1e5, alpha3=1e2, alpha4=1e-1,
                                                plot_flag=True, print_message=True)