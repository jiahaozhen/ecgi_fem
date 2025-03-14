import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import pyvista
import multiprocessing

sys.path.append('.')
from utils.helper_function import delta_tau, delta_deri_tau, compute_error, eval_function

def resting_ischemia_inversion(mesh_file, d_data, 
                                     gdim=3, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
                                     ischemia_potential=-60, normal_potential=-90, 
                                     tau=1, v_exact_file='3d/data/v.npy',
                                     multi_flag=True, plot_flag=False, exact_flag=False):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    sub_node_num = subdomain_ventricle.topology.index_map(tdim - 3).size_local

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
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

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
    form_loss = form(loss_element)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    j_p = (ischemia_potential - normal_potential) * delta_deri_phi * u2 * dot(grad(w), dot(Mi, grad(phi))) * dx2 \
            + (ischemia_potential - normal_potential) * delta_phi * dot(grad(w), dot(Mi, grad(u2))) * dx2
    form_J_p = form(j_p, entity_maps=entity_map)
    J_p = create_vector(form_J_p)

    # initial phi
    phi_0 = np.full(phi.x.array.shape, tau/2)
    phi.x.array[:] = phi_0
    delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
    
    # exact solution
    if exact_flag == True:
        v_exact = Function(V2)
        v_exact.x.array[:] = np.load(v_exact_file)

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
    while (True):
        delta_deri_phi.x.array[:] = delta_deri_tau(phi.x.array, tau)

        # cost function
        loss = assemble_scalar(form_loss)
        loss_per_iter.append(loss)
        if exact_flag == True:
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
        
        # print('iteration:', k)
        # print('loss:', loss)
        # print(k, 'J_p', np.linalg.norm(J_p.array))
        if exact_flag == True:
            print('center of mass error:', compute_error(v_exact, phi)[0])
        # check if the condition is satisfied
        if (k > 1e2 or np.linalg.norm(J_p.array) < 1e-1):
            break
        k = k + 1

        # updata p from partial derivative
        phi_v = phi.x.array[:].copy()
        dir_p = -J_p.array.copy()
        # origin value
        alpha = 1
        gamma = 0.5
        c = 0.1
        step_search = 0
        # print("start line search")
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
            loss_new = assemble_scalar(form_loss)
            loss_cmp = loss_new - (loss + c * alpha * J_p.array.dot(dir_p))
            alpha = gamma * alpha
            step_search = step_search + 1
            if (step_search > 20 or loss_cmp < 0):
                break
        # print("end line search")

    if plot_flag == False:
        return phi.x.array
    np.save('3d/data/phi_result.npy', phi.x.array)

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

    marker = Function(V2)
    marker_val = np.zeros(sub_node_num)
    marker_val[phi.x.array < 0] = 1
    marker.x.array[:] = marker_val

    def plot_f_on_subdomain(f, subdomain, title):
        grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
        grid.point_data["f"] = eval_function(f, subdomain.geometry.x)
        grid.set_active_scalars("f")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.add_title(title)
        plotter.view_xy()
        plotter.add_axes()
        plotter.show()

    p1 = multiprocessing.Process(target=plot_f_on_subdomain, args=(marker, subdomain_ventricle, 'ischemia_result'))
    p1.start()
    if (exact_flag == True):
        p2 = multiprocessing.Process(target=plot_f_on_subdomain, args=(v_exact, subdomain_ventricle, 'ischemia_exact'))
        p2.start()
        p2.join()
    p1.join()
    return phi

if __name__ == '__main__':
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    # d = np.load('3d/data/d.npy')
    d = np.load('3d/data/u_data_reaction_diffusion.npy')[0]
    resting_ischemia_inversion(mesh_file, d, plot_flag=True, exact_flag=False)