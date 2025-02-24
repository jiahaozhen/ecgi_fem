import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary, meshtags
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, compute_error, petsc2array, eval_function

# mesh of Body
file = "3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(tdim - 3).size_local
sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

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

# paramter
a1 = -90 # no active no ischemia
a2 = -60 # no active ischemia
a3 = 20 # active no ischemia
a4 = -10 # active ischemia
tau = 1

# phi G_phi delta_phi delta_deri_phi
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
# define d's value on the boundary
d_all_time = np.load(file='3d/data/d_all.npy')
time_total = np.shape(d_all_time)[0]

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
b_u_element = (a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 +\
        (a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2 +\
        (a3 - a4) * delta_phi_1 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 +\
        (a2 - a4) * delta_phi_2 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2
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
form_loss_1 = form(loss_element_1)

# vector b_w
b_w_element = u1 * (u - d) * ds
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction
u2 = TestFunction(V2)
j_p = -(a1-a2-a3+a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 +\
        -(a1-a2) * delta_deri_phi_1 * G_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 +\
        -(a1-a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 +\
        -(a3-a4) * delta_deri_phi_1 * (1 - G_phi_2) * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 +\
        -(a3-a4) * delta_phi_1 * (1 - G_phi_2) * dot(grad(w), dot(Mi, grad(u2))) * dx2
form_J_p = form(j_p, entity_maps = entity_map)
J_p = create_vector(form_J_p)

j_q = -(a1-a2-a3+a4) * delta_phi_1 * delta_phi_2 * u2 * dot(grad(w), dot(Mi, grad(phi_1))) * dx2 +\
        -(a1-a3) * delta_deri_phi_2 * G_phi_1 * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 +\
        -(a1-a3) * delta_phi_2 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2 +\
        -(a2-a4) * delta_deri_phi_2 * (1 - G_phi_1) * u2 * dot(grad(w), dot(Mi, grad(phi_2))) * dx2 +\
        -(a2-a4) * delta_phi_2 * (1 - G_phi_1) * dot(grad(w), dot(Mi, grad(u2))) * dx2
form_J_q = form(j_q, entity_maps = entity_map)
J_q = create_vector(form_J_q)

# step 1
step1 = np.full(sub_node_num, 0.5)
step1[sub_domain_boundary] = 0.1
step1 = np.diag(step1)
# step 2
step2 = np.full(sub_node_num, 0.5)
step2[sub_domain_boundary] = 0.1
step2 = np.diag(step2)

#initial phi_1 phi_2
v_exact = Function(V2)
v_exact.x.array[:] = np.load('3d/data/v.npy')
phi_1_0 = np.full(phi_1.x.array.shape, tau/2)
phi_2_0 = np.full(phi_2.x.array.shape, tau/2)
phi_1.x.array[:] = phi_1_0
phi_2.x.array[:] = phi_2_0

# phi result
phi_1_result = np.zeros((time_total, sub_node_num))
phi_2_result = np.zeros((time_total, sub_node_num))

cm_per_timeframe = []
loss_per_timeframe = []

for timeframe in range(time_total):
    k = 0
    d.x.array[:] = d_all_time[timeframe]
    while (k < 1e2):
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
        c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
        u.x.array[:] = u.x.array + c

        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)

        # compute partial derivative of p
        with J_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_p, form_J_p)
        # compute partial derivative of q
        with J_q.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_q, form_J_q)

        # check if the condition is satisfie
        loss = assemble_scalar(form_loss_1)
        if (np.linalg.norm(J_p.array) < 1e0 and np.linalg.norm(J_q.array) < 1e0):
             break
        
        # updata p from partial derivative
        phi_1.x.array[:] = phi_1.x.array - step1@(J_p.array/np.linalg.norm(J_p.array))
        phi_2.x.array[:] = phi_2.x.array - step2@(J_q.array/np.linalg.norm(J_q.array))
        k = k + 1
    print('timeframe:', timeframe)
    print('end at', k, 'iteration')
    print('J_p:', np.linalg.norm(J_p.array))
    print('J_q:', np.linalg.norm(J_q.array))
    print('loss:', loss)
    phi_1_result[timeframe] = phi_1.x.array
    phi_2_result[timeframe] = phi_2.x.array
    cm_per_timeframe.append(compute_error(v_exact, phi_1))
    loss_per_timeframe.append(loss)
