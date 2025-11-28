from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from ufl import TestFunction, TrialFunction, dot, grad, Measure, derivative, sqrt, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import multiprocessing
from utils.error_metrics_tools import compute_error
from utils.transmembrane_potential_tools import delta_tau, delta_deri_tau
from utils.helper_function import find_vertex_with_neighbour_less_than_0
from utils.simulate_tools import build_M, build_Mi
from utils.visualize_tools import plot_f_on_domain, plot_loss_and_cm

def resting_ischemia_inversion(mesh_file, d_data, v_data=None,
                               gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
                               tau=10, alpha1=1e-2, 
                               ischemia_potential=-80, normal_potential=-90, 
                               multi_flag=True, plot_flag=False, 
                               print_message=False, transmural_flag=False):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    sub_node_num = subdomain_ventricle.topology.index_map(tdim - 3).size_local

    # function space
    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

    M = build_M(domain, cell_markers, condition=None, 
                sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t, 
                multi_flag=multi_flag)
    Mi = build_Mi(subdomain_ventricle, condition=None, 
                  sigma_i=sigma_i)

    # phi delta_phi delta_deri_phi
    phi = Function(V2)
    delta_phi = Function(V2)
    delta_deri_phi = Function(V2)

    u = Function(V1)
    w = Function(V1)
    # function d
    d = Function(V1)
    # define d's value on the boundary
    d.x.array[:] = d_data

    # matrix A_u
    u1 = TestFunction(V1)
    v1 = TrialFunction(V1)
    dx1 = Measure("dx", domain=domain)
    a_element = dot(grad(u1), dot(M, grad(v1))) * dx1
    bilinear_form_a = form(a_element)
    A_u = assemble_matrix(bilinear_form_a)
    A_u.assemble()
    solver = PETSc.KSP().create()
    solver.setOperators(A_u)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)

    # vector b_u
    dx2 = Measure("dx",domain=subdomain_ventricle)
    b_u_element = (ischemia_potential - normal_potential) * delta_phi * dot(grad(u1), dot(Mi, grad(phi))) * dx2
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
    reg_element = alpha1 * delta_phi * sqrt(inner(grad(phi), grad(phi)) + 1e-8) * dx2
    form_loss = form(loss_element)
    form_reg = form(reg_element)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    j_p = (ischemia_potential - normal_potential) * delta_deri_phi * u2 * dot(grad(w), dot(Mi, grad(phi))) * dx2 \
            + (ischemia_potential - normal_potential) * delta_phi * dot(grad(w), dot(Mi, grad(u2))) * dx2
    reg_p = alpha1 * derivative(delta_phi * sqrt(inner(grad(phi), grad(phi)) + 1e-8) * dx2, phi, u2)
    form_J_p = form(j_p, entity_maps=entity_map)
    form_Reg_p = form(reg_p, entity_maps=entity_map)
    J_p = create_vector(form_J_p)
    Reg_p = create_vector(form_Reg_p)

    # initial phi
    phi_0 = np.full(phi.x.array.shape, tau/2)
    # phi_0 = np.where(v_data == -80, -tau/2, tau/2)
    # phi_0 = get_epi_endo_marker(V2) * tau / 1e2
    phi.x.array[:] = phi_0
    delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
    
    # exact solution
    v_exact = Function(V2)
    v_exact.x.array[:] = v_data

    # get u from p
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1)/assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    loss_per_iter = []
    cm_cmp_per_iter = []

    k = 0
    total_iter = 2e2
    while (True):
        delta_deri_phi.x.array[:] = delta_deri_tau(phi.x.array, tau)

        # cost function
        loss = assemble_scalar(form_loss) + assemble_scalar(form_reg)
        loss_per_iter.append(loss)
        cm_cmp_per_iter.append(compute_error(v_exact, phi)[0])

        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)
        # compute partial derivative of p from w
        with J_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_p, form_J_p)
        with Reg_p.localForm() as loc_R:
                loc_R.set(0)
        assemble_vector(Reg_p, form_Reg_p)
        J_p = J_p + Reg_p
        if print_message == True:
            print('iteration:', k)
            print('loss_residual:', assemble_scalar(form_loss))
            print('loss_reg:', assemble_scalar(form_reg))
            print('J_p', np.linalg.norm(J_p.array))
            print('center of mass error:', compute_error(v_exact, phi)[0])
        # check if the condition is satisfied
        if (k > total_iter or np.linalg.norm(J_p.array) < 1e-1):
            break
        k = k + 1

        # updata p from partial derivative
        dir_p = -J_p.array.copy()
        phi_v = phi.x.array[:].copy()
        
        # Barzilai-Borwein Method
        # J_p_current = J_p.array.copy()
        # if k == 1:
        #     step = 1e-3
        # else:
        #     s_k = phi_v - phi_v_prev
        #     y_k = J_p_current - J_p_prev
        #     step = np.dot(s_k, s_k) / np.dot(y_k, s_k)
        # phi.x.array[:] = phi_v + step * dir_p
        # phi_v_prev = phi_v.copy()
        # J_p_prev = J_p_current.copy()
        # # compute u
        # delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
        # with b_u.localForm() as loc_b:
        #     loc_b.set(0)
        # assemble_vector(b_u, linear_form_b_u)
        # solver.solve(b_u, u.vector)
        # # adjust u
        # adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
        # u.x.array[:] = u.x.array + adjustment
        
        # Armijo Method
        alpha = 1
        gamma = 0.8
        c = 0.1
        step_search = 0
        while(True):
            # adjust p
            phi.x.array[:] = phi_v + alpha * dir_p
            # compute u
            delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment

            # compute loss
            loss_new = assemble_scalar(form_loss) + assemble_scalar(form_reg)
            loss_cmp = loss_new - (loss + c * alpha * J_p.array.dot(dir_p))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 100 or loss_cmp < 0):
                if transmural_flag == True:
                    # for p < 0, make its neighbor smaller
                    neighbour_idx, _ = find_vertex_with_neighbour_less_than_0(subdomain_ventricle, phi) 
                    # make them smaller
                    phi.x.array[neighbour_idx] = np.where(phi.x.array[neighbour_idx] >= 0, 
                                                          phi.x.array[neighbour_idx] - tau / total_iter, 
                                                          phi.x.array[neighbour_idx])
                    # compute u
                    delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
                    with b_u.localForm() as loc_b:
                        loc_b.set(0)
                    assemble_vector(b_u, linear_form_b_u)
                    solver.solve(b_u, u.vector)
                    # adjust u
                    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                    u.x.array[:] = u.x.array + adjustment
                break

    if not plot_flag:
        return phi, assemble_scalar(form_loss), assemble_scalar(form_reg)

    marker = Function(V2)
    marker_val = np.zeros(sub_node_num)
    marker_val[phi.x.array < 0] = 1
    marker.x.array[:] = marker_val

    marker_exact = Function(V2)
    marker_exact.x.array[:] = np.where(v_exact.x.array == ischemia_potential, 1, 0)

    p1 = multiprocessing.Process(target=plot_f_on_domain, args=(subdomain_ventricle, marker, 'ischemia_result'))
    p2 = multiprocessing.Process(target=plot_f_on_domain, args=(subdomain_ventricle, marker_exact, 'ischemia_exact'))
    p3 = multiprocessing.Process(target=plot_loss_and_cm, args=(loss_per_iter, cm_cmp_per_iter))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    return phi, assemble_scalar(form_loss), assemble_scalar(form_reg)