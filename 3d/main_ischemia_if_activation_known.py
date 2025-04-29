import sys
import time

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, dot, grad, Measure, derivative, sqrt, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, eval_function, compute_error_phi, find_vertex_with_neighbour_less_than_0

def ischemia_inversion(mesh_file, d_data, v_data, alpha1, time_sequence,
                       phi_1_exact, phi_2_exact, tau=10, 
                       gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                       a1=-90, a2=-60, a3=10, a4=-20,
                       plot_flag=False, print_message=False) -> Function:
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
    v = Function(V2)

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
    reg_element = alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2

    form_loss = form(loss_element)
    form_reg = form(reg_element)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    residual_p = (- (a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
                  - (a1 - a2) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a1 - a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2
                  - (a3 - a4) * delta_deri_phi_1 * (1 - G_phi_2) * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
                  - (a3 - a4) * delta_phi_1 * (1 - G_phi_2) * dot(grad(w), dot(Mi, grad(u2))) * dx2)
    reg_p = derivative(alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2, phi_1, u2)
    form_Residual_p = form(residual_p, entity_maps=entity_map)
    form_Reg_p = form(reg_p, entity_maps=entity_map)
    J_p = create_vector(form_Residual_p)
    Reg_p = create_vector(form_Reg_p)

    phi_2_result = phi_2_exact.copy()

    phi_1.x.array[:] = np.full(phi_1.x.array.shape, tau/2)
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)
    
    # fix phi_2 for phi_1
    print('start computing phi_1 with phi_2 fixed')

    def compute_u_from_phi_1(phi_1: Function):
        G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        u_array = np.zeros((d_data.shape[0], domain.topology.index_map(0).size_local))
        for timeframe in time_sequence:
            d.x.array[:] = d_data[timeframe]
            #  TODO: \phi_2 initial with noise
            phi_2.x.array[:] = phi_2_result[timeframe]
            # phi_2.x.array[:] = np.where(phi_2_result[timeframe] < 0, -tau/2, tau/2)
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
            delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
            # get u from p, q
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
        for timeframe in time_sequence:
            d.x.array[:] = d_data[timeframe]
            u.x.array[:] = u_array[timeframe]
            phi_2.x.array[:] = phi_2_result[timeframe]
            # phi_2.x.array[:] = np.where(phi_2_result[timeframe] < 0, -tau/2, tau/2)
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
            J_p_array = J_p_array + J_p.array.copy() + Reg_p.array.copy()
        return J_p_array
    
    def compute_loss_from_phi_1(phi_1: Function, u_array: np.ndarray):
        if u_array is None:
            u_array = compute_u_from_phi_1(phi_1)
        loss_residual = 0
        loss_reg = 0
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        for timeframe in time_sequence:    
            d.x.array[:] = d_data[timeframe]
            u.x.array[:] = u_array[timeframe]
            loss_residual = loss_residual + assemble_scalar(form_loss)
            loss_reg = loss_reg + assemble_scalar(form_reg)
        return loss_residual, loss_reg
    
    total_iter = 200
    loss_per_iter = []
    cm_per_iter = []
    k = 0
    u_array = compute_u_from_phi_1(phi_1)
    start_time = time.time()
    while (True):
        loss_residual, loss_reg = compute_loss_from_phi_1(phi_1, u_array)
        loss = loss_residual + loss_reg
        loss_per_iter.append(loss)
        cm1 = compute_error_phi(phi_1.x.array, phi_1_exact[0], V2)
        cm_per_iter.append(cm1)
        J_p_array =  compute_Jp_from_phi_1(phi_1, u_array)
        end_time = time.time()

        if print_message:
            print('iteration:', k)
            print('loss_residual:', loss_residual)
            print('loss_reg:', loss_reg)
            print('J_p:', np.linalg.norm(J_p_array))
            print('center of mass error:', cm1)
            print('cost', end_time - start_time, 'seconds')
        if (k > total_iter or np.linalg.norm(J_p_array) < 1e-1):
            break
        k = k + 1
        start_time = time.time()

        phi_1_v = phi_1.x.array[:].copy()
        dir_p = -J_p_array.copy()
        alpha = 1
        gamma = 0.8
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
                # for p < 0, make its neighbor smaller
                neighbour_idx, _ = find_vertex_with_neighbour_less_than_0(subdomain_ventricle, phi_1) 
                # make them smaller
                phi_1.x.array[neighbour_idx] = np.where(phi_1.x.array[neighbour_idx] >= 0, 
                                                        phi_1.x.array[neighbour_idx] - tau / total_iter, 
                                                        phi_1.x.array[neighbour_idx])
                # compute u
                u_array = compute_u_from_phi_1(phi_1)             
                break
    
    if plot_flag == False:
        return phi_1

    def plot_loss_and_error():
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.plot(loss_per_iter)
        plt.title('cost functional')
        plt.xlabel('iteration')
        plt.subplot(1, 2, 2)
        plt.plot(cm_per_iter)
        plt.title('error in center of mass')
        plt.xlabel('iteration')
        plt.show()

    marker_result = Function(V2)
    marker_result.x.array[:] = np.where(phi_1.x.array < 0, 1, 0)

    marker_exact = Function(V2)
    marker_exact.x.array[:] = np.where(phi_1_exact[0] < 0, 1, 0)

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

    p1 = multiprocessing.Process(target=plot_f_on_subdomain, args=(marker_result, subdomain_ventricle, 'ischemia_result'))
    p2 = multiprocessing.Process(target=plot_f_on_subdomain, args=(marker_exact, subdomain_ventricle, 'ischemia_exact'))
    p3 = multiprocessing.Process(target=plot_loss_and_error)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    return phi_1


if __name__ == '__main__':
    mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    v_data_file = '3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy'
    d_data_file = '3d/data/u_data_reaction_diffusion_ischemia_data_argument.npy'
    phi_1_file = "3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
    phi_2_file = "3d/data/phi_2_data_reaction_diffusion_ischemia.npy"
    v_data = np.load(v_data_file)
    d_data = np.load(d_data_file)
    phi_1_exact = np.load(phi_1_file)
    phi_2_exact = np.load(phi_2_file)
    time_sequence = np.arange(600, 601, 1)
    phi_1 = ischemia_inversion(mesh_file=mesh_file, d_data=d_data, v_data=v_data, 
                               phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                               time_sequence=time_sequence,
                               gdim=3, tau=10, alpha1=1e-100,
                               plot_flag=True, print_message=True)