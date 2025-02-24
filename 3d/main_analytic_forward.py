import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.fem import functionspace, Function, form, assemble_scalar
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from ufl import TrialFunction, TestFunction, grad, dot, Measure
from mpi4py import MPI
from dolfinx.plot import vtk_mesh
from petsc4py import PETSc
import numpy as np
import pyvista

sys.path.append('.')
from utils.helper_function import submesh_node_index, compute_normal, eval_function, assign_function
from utils.analytic_tool import calculate_ue_gradients, calculate_ut_gradients, calculate_ue_values, calculate_ut_values

sigma_i = 0.2
sigma_e = 0.8
sigma_t = 0.8
source_point=[0.55,-0.55,0.55]

def compute_g(domain):
    tdim = domain.topology.dim
    # normal
    nh_values = compute_normal(domain)

    # ut ue
    boundary_index = locate_entities_boundary(domain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    coords = domain.geometry.x[boundary_index]
    ut_gradient_values = calculate_ut_gradients(coords, source_point).T
    ue_gradient_values = calculate_ue_gradients(coords, source_point).T

    # g
    g = -(sigma_i+sigma_e)/sigma_i * np.sum(ue_gradient_values*nh_values, axis=1)\
        + sigma_t/sigma_i * np.sum(ut_gradient_values*nh_values, axis=1)\
        + 2*(sigma_i+sigma_e)/sigma_i * np.sum(coords*nh_values, axis=1)
    
    return g

def main():
    domain, cell_markers, _ = gmshio.read_from_msh('3d/data/heart_torso.msh', MPI.COMM_WORLD, gdim=3)
    tdim = domain.topology.dim

    subdomain, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain, ("Lagrange", 1))

    # tmp v
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    g = Function(V2)
    boundary_index = locate_entities_boundary(domain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    subboundary_index = locate_entities_boundary(subdomain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    sub_to_parent_index = submesh_node_index(domain, cell_markers, 2)
    g_val = np.zeros((len(subdomain.geometry.x)))
    g_val[subboundary_index] = compute_g(subdomain)
    assign_function(g, np.arange(len(subdomain.geometry.x)), g_val)
    dx = Measure('dx', domain=subdomain)
    ds = Measure('ds', domain=subdomain)
    a = dot(grad(u2), grad(v2)) * dx
    L = g * v2 * ds
    problem = LinearProblem(a, L, petsc_options={"ksp_type": "lsqr", "pc_type": "ilu"})
    v_f = problem.solve()

    v = Function(V2)
    v.x.array[:] = v_f.x.array - np.mean(v_f.x.array) 
    v.x.array[:] = v.x.array - 5 * np.sum(V2.tabulate_dof_coordinates()*V2.tabulate_dof_coordinates(), axis=1)
    v_exact = -calculate_ut_values(subdomain.geometry.x, source_point)
    v_exact = v_exact - np.mean(v_exact)
    v_exact = v_exact - 5 * np.sum(subdomain.geometry.x*subdomain.geometry.x, axis=1)

    # boundary gt
    gt = Function(V1)
    coords = domain.geometry.x[boundary_index]
    ut_gradient_values = calculate_ut_gradients(coords, source_point).T
    nt_values = compute_normal(domain)
    gt.x.array[boundary_index] = np.sum(nt_values*ut_gradient_values, axis=1) * sigma_t

    # forward u
    M = Function(V1)
    M_value = np.full_like(M.x.array, sigma_t)
    M_value[sub_to_parent_index] = sigma_i + sigma_e
    M.x.array[:] = M_value

    u = Function(V1)
    # A u = b
    # matrix A
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    dx1 = Measure("dx", domain = domain)
    a_element = M * dot(grad(u1), grad(v1)) * dx1
    bilinear_form_a = form(a_element)
    A = assemble_matrix(bilinear_form_a)
    A.assemble()

    # b
    dx2 = Measure("dx", domain = subdomain)
    ds = Measure("ds", domain = domain)
    b_element_1 = -sigma_i * dot(grad(v1), grad(v)) * dx2 
    b_element_2 = v1 * gt * ds
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b_1 = form(b_element_1, entity_maps = entity_map)
    linear_form_b_2 = form(b_element_2, entity_maps = entity_map)
    b = assemble_vector(linear_form_b_1) + assemble_vector(linear_form_b_2)

    solver = PETSc.KSP().create()
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.LSQR)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.solve(b, u.vector)

    # u_exact
    u_exact = Function(V1)
    u_exact.x.array[:] = calculate_ut_values(domain.geometry.x, source_point)
    u_exact.x.array[sub_to_parent_index] = calculate_ue_values(subdomain.geometry.x, source_point)

    # adjust result
    ds = Measure('ds', domain = domain)
    c1_element = (u_exact - u) * ds
    c2_element = 1 * ds
    form_c1 = form(c1_element)
    form_c2 = form(c2_element)
    c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + c

    print('u relative error:', np.linalg.norm(u.x.array - u_exact.x.array) / np.linalg.norm(u_exact.x.array))

    # plot
    plotter_v = pyvista.Plotter(shape=(1,3))

    plotter_v.subplot(0, 0)
    grid_v_exact = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    v_exact_boundary = v_exact
    mask = ~np.isin(np.arange(len(v_exact_boundary)), subboundary_index)
    v_exact_boundary[mask] = v_exact_boundary[subboundary_index[0]]
    grid_v_exact.point_data["v_exact"] = v_exact_boundary
    grid_v_exact.set_active_scalars("v_exact")
    plotter_v.add_mesh(grid_v_exact, show_edges=True)
    plotter_v.view_xz()
    plotter_v.add_axes()
    plotter_v.add_title("transmembrane potential exact (v_exact)", font_size=9)

    plotter_v.subplot(0, 1)
    grid_v = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    v_boundary = eval_function(v, subdomain.geometry.x).squeeze()
    mask = ~np.isin(np.arange(len(v_boundary)), subboundary_index)
    v_boundary[mask] = v_boundary[subboundary_index[0]]
    grid_v.point_data["v"] = v_boundary
    grid_v.set_active_scalars("v")
    plotter_v.add_mesh(grid_v, show_edges=True)
    plotter_v.view_xz()
    plotter_v.add_axes()
    plotter_v.add_title("transmembrane potential (v)", font_size=9)

    plotter_v.subplot(0, 2)
    grid_v_error = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    grid_v_error.point_data["v_error"] = v_boundary - v_exact_boundary
    grid_v_error.set_active_scalars("v_error")
    plotter_v.add_mesh(grid_v_error, show_edges=True)
    plotter_v.view_xz()
    plotter_v.add_axes()
    plotter_v.add_title("transmembrane potential error (v_error)", font_size=9)

    plotter_v.show()

    plotter = pyvista.Plotter(shape=(2, 3))
    plotter.subplot(0, 0)
    grid_ue_exact = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    ue_exact_boundary = eval_function(u_exact, subdomain.geometry.x)
    mask = ~np.isin(np.arange(len(ue_exact_boundary)), subboundary_index)
    ue_exact_boundary[mask] = ue_exact_boundary[subboundary_index[0]]
    grid_ue_exact.point_data["ue_exact"] = ue_exact_boundary
    grid_ue_exact.set_active_scalars("ue_exact")
    plotter.add_mesh(grid_ue_exact, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Heart potential exact (ue_exact)", font_size=9)

    plotter.subplot(1, 0)
    grid_ut_exact = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    ut_exact_boundary = eval_function(u_exact, domain.geometry.x)
    mask = ~np.isin(np.arange(len(ut_exact_boundary)), boundary_index)
    ut_exact_boundary[mask] = ut_exact_boundary[boundary_index[0]]
    grid_ut_exact.point_data["ut_exact"] = ut_exact_boundary
    grid_ut_exact.set_active_scalars("ut_exact")
    plotter.add_mesh(grid_ut_exact, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Torso potential exact (ut_exact)", font_size=9)

    plotter.subplot(0, 1)
    grid_ue = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    ue_boundary = eval_function(u, subdomain.geometry.x)
    mask = ~np.isin(np.arange(len(ue_boundary)), subboundary_index)
    ue_boundary[mask] = ue_boundary[subboundary_index[0]]
    grid_ue.point_data["ue"] = ue_boundary
    grid_ue.set_active_scalars("ue")
    plotter.add_mesh(grid_ue, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Heart potential (ue)", font_size=9)

    plotter.subplot(1, 1)
    grid_ut = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    ut_boundary = eval_function(u, domain.geometry.x)
    mask = ~np.isin(np.arange(len(ut_boundary)), boundary_index)
    ut_boundary[mask] = ut_boundary[boundary_index[0]]
    grid_ut.point_data["ut"] = ut_boundary
    grid_ut.set_active_scalars("ut")
    plotter.add_mesh(grid_ut, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Torso potential (ut)", font_size=9)

    plotter.subplot(0, 2)
    grid_ue_error = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    grid_ue_error.point_data["ue_error"] = ue_boundary - ue_exact_boundary
    grid_ue_error.set_active_scalars("ue_error")
    plotter.add_mesh(grid_ue_error, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Heart potential error (ue_error)", font_size=9)

    plotter.subplot(1, 2)
    grid_ut_error = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    grid_ut_error.point_data["ut_error"] = ut_boundary - ut_exact_boundary
    grid_ut_error.set_active_scalars("ut_error")
    plotter.add_mesh(grid_ut_error, show_edges=True)
    plotter.view_xz()
    plotter.add_axes()
    plotter.add_title("Torso potential error (ut_error)", font_size=9)

    plotter.show()

if __name__ == '__main__':
    main()