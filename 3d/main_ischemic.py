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

sys.path.append('.')
from utils.helper_function import delta_tau, delta_deri_tau, compute_error, petsc2array, eval_function

# mesh of Body
file = "3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(tdim - 3).size_local

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
V3 = functionspace(subdomain_ventricle, ("Lagrange", 1, (tdim, )))

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

V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
M.interpolate(rho3, cell_markers.find(3))
M.interpolate(rho4, cell_markers.find(4))
Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim)*sigma_i))

# parameter a2 a1 tau
a1 = -60
a2 = -90
tau = 1
regularization_parameter = 0
# phi delta_phi delta_deri_phi
phi = Function(V2)
delta_phi = Function(V2)
delta_deri_phi = Function(V2)
u = Function(V1)
w = Function(V1)
# function d
d = Function(V1)
# define d's value on the boundary
d.x.array[:] = np.load(file='3d/data/d.npy')

LTL_matrix = np.load('3d/data/LTL_matrix.npy')
# LTL_matrix = np.load('3d/data/LTL_integral.npy')

# matrix A_u
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx", domain = domain)
a_element = dot(grad(u1), dot(M, grad(v1))) * dx1
bilinear_form_a = form(a_element)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()
solver = PETSc.KSP().create()
solver.setOperators(A_u)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# vector b_u
dx2 = Measure("dx",domain = subdomain_ventricle)
b_u_element = (a1 - a2) * delta_phi * dot(grad(u1), dot(Mi, grad(phi))) * dx2
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
# loss_element_2 = 0.5 * regularization_parameter * (div(grad_phi)) ** 2 * dx2
form_loss_1 = form(loss_element_1)
# form_loss_2 = form(loss_element_2)

# vector b_w
b_w_element = u1 * (u - d) * ds
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction
u2 = TestFunction(V2)
j_p = (a1 - a2) * delta_deri_phi * u2 * dot(grad(w), dot(Mi, grad(phi))) * dx2\
        + (a1 - a2) * delta_phi * dot(grad(w), dot(Mi, grad(u2))) * dx2\
        # + regularization_parameter * div(grad_phi) * div(grad(u2)) * dx2
form_J_p = form(j_p, entity_maps = entity_map)
J_p = create_vector(form_J_p)

# initial phi
v_exact = Function(V2)
v_exact.x.array[:] = np.load('3d/data/v.npy')
phi_0 = np.full(phi.x.array.shape, tau/2)
phi.x.array[:] = phi_0

# step B
# step = np.full(sub_node_num, 0.8)
# sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
# step[sub_domain_boundary] = 0.4
# step = np.diag(step)

delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
# get u from p
with b_u.localForm() as loc_b:
    loc_b.set(0)
assemble_vector(b_u, linear_form_b_u)
solver.solve(b_u, u.vector)
# adjust u
c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
u.x.array[:] = u.x.array + c

loss_per_iter = []
cm_cmp_per_iter = []

k = 0
while (k < 1e2):
    cm_cmp_per_iter.append(compute_error(v_exact, phi)[0]/134.04)
    delta_deri_phi.x.array[:] = delta_deri_tau(phi.x.array, tau)

    # cost function
    loss1 = assemble_scalar(form_loss_1)
    # loss2 = assemble_scalar(form_loss_2)
    # loss3 = regularization_parameter * (phi.x.array[:].copy().T@LTL_matrix@phi.x.array[:].copy())
    loss = loss1
    loss_per_iter.append(loss)
    print(k, 'J', loss)
    # print(k, 'regularization_term1', loss2)
    # print(k, 'regularization_term2', loss3)

    # get w from u
    with b_w.localForm() as loc_w:
        loc_w.set(0)
    assemble_vector(b_w, linear_form_b_w)
    solver.solve(b_w, w.vector)

    # compute partial derivative of p from w
    with J_p.localForm() as loc_J:
        loc_J.set(0)
    assemble_vector(J_p, form_J_p)
    phi_v = phi.x.array[:].copy()
    J_p_array = J_p.array.copy()
    # J_p_array = J_p_array + 2 * regularization_parameter * LTL_matrix @ phi_v
    print(k, 'J_p', np.linalg.norm(J_p_array))

    # condition: J_p
    if (np.linalg.norm(J_p_array) < 1e-1):
        break
    # updata p from partial derivative
    # origin value
    alpha = 1
    gamma = 0.5
    c = 0.1
    print("start line search")
    while(True):
        # adjust p
        phi.x.array[:] = phi_v - alpha * J_p_array
        # compute u
        delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
        with b_u.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b_u, linear_form_b_u)
        solver.solve(b_u, u.vector)
        # adjust u
        c = assemble_scalar(form_c1) / assemble_scalar(form_c2)
        u.x.array[:] = u.x.array + c
        # compute loss
        J = assemble_scalar(form_loss_1) 
        # J = J + assemble_scalar(form_loss_2)
        phi_val_on_submesh = eval_function(phi, subdomain_ventricle.geometry.x)
        # J = J + regularization_parameter * (phi_val_on_submesh.T@LTL_matrix@phi_val_on_submesh)
        J_cmp = J - (loss - c * alpha * np.linalg.norm(J_p_array)**2)
        if (J_cmp < 1e-2):
            break
        alpha = gamma * alpha
    k = k + 1

print(compute_error(v_exact, phi))

np.save('3d/data/phi_result.npy', phi.x.array[:])

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

grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid1.point_data["v"] = eval_function(v_exact, subdomain_ventricle.geometry.x)
grid1.set_active_scalars("v")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid2.point_data["marker"] = eval_function(marker, subdomain_ventricle.geometry.x)
grid2.set_active_scalars("marker")
grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid3.point_data["phi"] = eval_function(phi, subdomain_ventricle.geometry.x)
grid3.set_active_scalars("phi")
grid = (grid1, grid2, grid3)

plotter = pyvista.Plotter(shape=(1, 2))
for i in range(2):
    plotter.subplot(0, i)
    plotter.add_mesh(grid[i], show_edges=True)
    plotter.view_xy()
    plotter.add_axes()
plotter.show()