import sys

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
import matplotlib.pyplot as plt
import pyvista
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, compute_error_phi, eval_function, find_vertex_with_neighbour_less_than_0

# if ischemia zone is known, find activation zone
def activation_inversion_with_ischemia(mesh_file, d_data, v_data, phi_1_exact, phi_2_exact, timeframe,
                               gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
                               a1=-90, a2=-60, a3=10, a4=-20, tau=10, 
                               multi_flag=True):
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
        if multi_flag == True:
            M.interpolate(rho3, cell_markers.find(3))
        else:
            M.interpolate(rho1, cell_markers.find(3))
    if cell_markers.find(4).any():
        if multi_flag == True:
            M.interpolate(rho4, cell_markers.find(4))
        else:
            M.interpolate(rho1, cell_markers.find(4))
    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))

    alpha1 = 1e-2
    alpha2 = 1e1
    # phi delta_phi delta_deri_phi
    phi_1 = Function(V2)
    G_phi_1 = Function(V2)
    delta_phi_1 = Function(V2)
    delta_deri_phi_1 = Function(V2)
    phi_2 = Function(V2)
    G_phi_2 = Function(V2)
    delta_phi_2 = Function(V2)
    delta_deri_phi_2 = Function(V2)
    # phi_2_est = Function(V2)
    # phi_2_mono = Function(V2)
    # phi_2_I = Function(V2)

    u = Function(V1)
    w = Function(V1)
    v = Function(V2)
    # function d
    d = Function(V1)
    # define d's value on the boundary
    d.x.array[:] = d_data[timeframe]

    # matrix A_u
    u1 = TestFunction(V1)
    v1 = TrialFunction(V1)
    dx1 = Measure("dx", domain=domain)
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
    reg_element_1 = alpha1 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2
    # reg_element_2 = 0.5 * alpha2 * phi_2_mono ** 2 * dx2
    form_loss = form(loss_element)
    # form_reg = form(reg_element_1 + reg_element_2)
    form_reg = form(reg_element_1)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    j_q = (- (a1 - a2 - a3 + a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 
           - (a1 - a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
           - (a1 - a3) * delta_phi_2 * G_phi_1 * dot(grad(w), dot(Mi, grad(u2))) * dx2 
           - (a2 - a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 
           - (a2 - a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2)
    reg_q_1 = alpha1 * derivative(delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2, phi_2, u2)
    # reg_q_2 = alpha2 * phi_2_mono * phi_2_I * u2 * dx2
    form_J_q = form(j_q, entity_maps=entity_map)
    # form_Reg_q = form(reg_q_1 + reg_q_2, entity_maps=entity_map)
    form_Reg_q = form(reg_q_1, entity_maps=entity_map)
    J_q = create_vector(form_J_q)
    Reg_q = create_vector(form_Reg_q)

    # initial phi
    # phi_1.x.array[:] = np.where(phi_1_exact[timeframe-1] < 0, -tau/2, tau/2)
    phi_1.x.array[:] = phi_1_exact[timeframe]
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
    delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)

    phi_0 = np.full(phi_2.x.array.shape, tau/2)
    # phi_0 = np.where(phi_2_exact[timeframe-1] < 0, -tau/2, tau/2)
    # phi_0 = phi_2_exact[timeframe]
    phi_2.x.array[:] = phi_0
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
    v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array + 
                    (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
    # get u from q
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    loss_per_iter = []
    cm_cmp_per_iter = []

    k = 0
    total_iter = 200
    while (True):
        delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
        delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)
        # phi_2_mono.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0, 
        #                                  phi_2.x.array - phi_2_est.x.array, 0)
        # phi_2_I.x.array[:] = np.where(phi_2.x.array - phi_2_est.x.array > 0, 1, 0)

        # cost function
        loss = assemble_scalar(form_loss) + assemble_scalar(form_reg)
        # loss = assemble_scalar(form_loss)
        loss_per_iter.append(loss)
        cm_cmp_per_iter.append(compute_error_phi(phi_2.x.array, phi_2_exact[timeframe], V2))

        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)
        # compute partial derivative of q from w
        with J_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_q, form_J_q)
        with Reg_q.localForm() as loc_R:
            loc_R.set(0)
        assemble_vector(Reg_q, form_Reg_q)
        print('iteration:', k)
        print('loss_residual:', assemble_scalar(form_loss))
        print('loss_reg:', assemble_scalar(form_reg))
        print('J_q', np.linalg.norm(J_q.array))
        print('Reg_q', np.linalg.norm(Reg_q.array))
        print('cm_cmp', cm_cmp_per_iter[-1])
        J_q = J_q + Reg_q
        # check if the condition is satisfied
        if (k > total_iter or np.linalg.norm(J_q.array) < 1e-1):
            break
        k = k + 1

        # updata q from partial derivative
        phi_v = phi_2.x.array[:].copy()
        dir_q = -J_q.array.copy()
        # origin value
        alpha = 1
        gamma = 0.8
        c = 0.1
        step_search = 0
        while(True):
            # adjust p
            phi_2.x.array[:] = phi_v + alpha * dir_q
            # compute u
            G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
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
            # loss_new = assemble_scalar(form_loss)
            loss_cmp = loss_new - (loss + c * alpha * J_q.array.dot(dir_q))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 100 or loss_cmp < 0):
                # for q < 0, make its neighbor smaller
                neighbor_idx, _ = find_vertex_with_neighbour_less_than_0(subdomain_ventricle, phi_2)
                # make them smaller
                phi_2.x.array[neighbor_idx] = np.where(phi_2.x.array[neighbor_idx] >= 0, 
                                                       phi_2.x.array[neighbor_idx] - tau / total_iter, 
                                                       phi_2.x.array[neighbor_idx])
                # compute u
                G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
                v.x.array[:] = ((a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array +
                                (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array))
                with b_u.localForm() as loc_b:
                    loc_b.set(0)
                assemble_vector(b_u, linear_form_b_u)
                solver.solve(b_u, u.vector)
                # adjust u
                adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                u.x.array[:] = u.x.array + adjustment
                break

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(loss_per_iter)
    plt.title('cost functional')
    plt.xlabel('iteration')
    plt.subplot(1, 2, 2)
    plt.plot(cm_cmp_per_iter)
    plt.title('error in center of mass')
    plt.xlabel('iteration')
    plt.show()

    marker_result = Function(V2)
    # marker_result.x.array[:] = np.where(phi_2.x.array < 0, 1, 0)
    marker_result.x.array[:] = phi_2.x.array

    marker_exact = Function(V2)
    # marker_exact.x.array[:] = np.where(phi_2_exact[timeframe] < 0, 1, 0)
    marker_exact.x.array[:] = phi_2_exact[timeframe]
    
    v_exact = Function(V2)
    v_exact.x.array[:] = v_data[timeframe]

    def plot_f_on_surface(f, title):
        grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
        f_boundary = eval_function(f, domain.geometry.x)
        boundary_index = locate_entities_boundary(domain, tdim-3, 
                                                  lambda x: np.full(x.shape[1], True, dtype=bool))
        mask = ~np.isin(np.arange(len(f_boundary)), boundary_index)
        f_boundary[mask] = f_boundary[boundary_index[0]]
        grid.point_data["f"] = f_boundary
        grid.set_active_scalars("f")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.add_title(title)
        plotter.view_yz()
        plotter.add_axes()
        plotter.show()

    def plot_f_on_subdomain(f, subdomain, title):
        grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
        grid.point_data["f"] = eval_function(f, subdomain.geometry.x)
        grid.set_active_scalars("f")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.add_title(title)
        plotter.view_yz()
        plotter.add_axes()
        plotter.show()

    p1 = multiprocessing.Process(target=plot_f_on_subdomain, args=(marker_result, subdomain_ventricle, 'activation_result'))
    p2 = multiprocessing.Process(target=plot_f_on_subdomain, args=(marker_exact, subdomain_ventricle, 'activation_exact'))
    p3 = multiprocessing.Process(target=plot_f_on_subdomain, args=(v_exact, subdomain_ventricle, 'v_exact'))
    p4 = multiprocessing.Process(target=plot_f_on_subdomain, args=(v, subdomain_ventricle, 'v_result'))
    p5 = multiprocessing.Process(target=plot_f_on_surface, args=(u, 'd_result'))
    p6 = multiprocessing.Process(target=plot_f_on_surface, args=(d, 'd_exact'))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    return phi_2.x.array

if __name__ == '__main__':
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    d_file = "3d/data/u_data_reaction_diffusion_ischemia_data_argument.npy"
    v_file = "3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy"
    phi_1_file = "3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
    phi_2_file = "3d/data/phi_2_data_reaction_diffusion_ischemia.npy"

    d = np.load(d_file)
    v = np.load(v_file)
    phi_1_exact = np.load(phi_1_file)
    phi_2_exact = np.load(phi_2_file)
    phi_2 = activation_inversion_with_ischemia(mesh_file, d, v, phi_1_exact, phi_2_exact, timeframe=100)