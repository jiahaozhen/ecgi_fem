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
from helper_function import delta_tau, delta_deri_tau, OuterBoundary1, OuterBoundary2, compare_CM
import matplotlib.pyplot as plt

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("2d/heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart 
#      M0 in Torso
def rho1(x):
    tensor = np.eye(2) * 2
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(2)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
V = functionspace(domain, ("DG", 0, (2,2)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
# M = Constant(domain, default_scalar_type(np.eye(tdim)))
Mi = Constant(domain, default_scalar_type(np.eye(tdim)))

# parameter a2 a1 tau
a1 = -60
a2 = -90
tau = 1
# phi delta_phi delta_deri_phi
phi = Function(V2)
delta_phi = Function(V2)
delta_deri_phi = Function(V2)
u = Function(V1)
w = Function(V1)
# function d
d = Function(V1)
# define d's value on the boundary
d.x.array[:] = np.load(file='2d/bsp_one_timeframe.npy')

#p0, p1, p2
p0 = Constant(subdomain, default_scalar_type(4))
p1 = Constant(subdomain, default_scalar_type(4))
p2 = Constant(subdomain, default_scalar_type(0.5))
vector0 = Constant(subdomain, default_scalar_type([1,0]))
vector1 = Constant(subdomain, default_scalar_type([0,1]))
x0 = Function(V2)
x0.interpolate(lambda x : x[0])
x1 = Function(V2)
x1.interpolate(lambda x : x[1])

class phi_model():
    def __init__(self, p0, p1, p2):
        self.x = p0
        self.y = p1
        self.r = p2
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2
        return dist - self.r ** 2

# matrix A_u
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx",domain=domain)
a_u = dot(grad(u1), dot(M, grad(v1))) * dx1
bilinear_form_a = form(a_u)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()
solver = PETSc.KSP().create()
solver.setOperators(A_u)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# vector b_u
dx2 = Measure("dx",domain=subdomain)
b_u_element = (a1 - a2) * delta_phi * dot(grad(u1), dot(Mi, grad(phi))) * dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b_u = form(b_u_element, entity_maps=entity_map)
b_u = create_vector(linear_form_b_u)

# outerBoundary
facets = locate_entities_boundary(domain, tdim - 1, OuterBoundary1)
facet_indices = np.array(facets, dtype=np.int32)
facet_values = np.full(len(facet_indices), 1, dtype=np.int32)
facet_tags = meshtags(domain, tdim - 1, facet_indices, facet_values)
ds_out = Measure('ds', domain=domain, subdomain_data=facet_tags, subdomain_id=1)

# scalar c
c1_element = (d-u) * ds_out
c2_element = 1 * ds_out
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element = 0.5 * (u - d)**2 * ds_out
form_loss = form(loss_element)

# vector b_w
b_w_element = u1 * (u - d) * ds_out
linear_form_b_w = form(b_w_element)
b_w = create_vector(linear_form_b_w)

# vector direction
j_p0_element = 2 * (a2 - a1) * dot(grad(w), dot(Mi, vector0 * delta_phi)) * dx2\
                + 2 * (a2 - a1) * (x0 - p0) * delta_deri_phi * dot(grad(w), dot(Mi, grad(phi))) * dx2
j_p1_element = 2 * (a2 - a1) * dot(grad(w), dot(Mi, vector1 * delta_phi)) * dx2\
                + 2 * (a2 - a1) * (x1 - p1) * delta_deri_phi * dot(grad(w), dot(Mi, grad(phi))) * dx2
j_p2_element = 2 * (a2 - a1) * p2 * delta_deri_phi * dot(grad(w), dot(Mi, grad(phi))) * dx2
form_J_p0 = form(j_p0_element, entity_maps=entity_map)
form_J_p1 = form(j_p1_element, entity_maps=entity_map)
form_J_p2 = form(j_p2_element, entity_maps=entity_map)

# initial p0, p1, p2
phi_exact = Function(V2)
phi_exact.x.array[:] = np.load(file='2d/phi_exact.npy')
phi_0 = Function(V2)
phi_0.interpolate(phi_model(p0, p1, p2))
# phi_0.x.array[:] = (subdomain.geometry.x[:,0]-p0)**2 + (subdomain.geometry.x[:,1]-p1)**2 - p2**2

loss_per_iter = []
cm_cmp_per_iter = []

phi.interpolate(phi_model(p0, p1, p2))
# phi.x.array[:] = (subdomain.geometry.x[:,0]-p0)**2 + (subdomain.geometry.x[:,1]-p1)**2 - p2**2
delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
# get u from p
with b_u.localForm() as loc_b:
    loc_b.set(0)
assemble_vector(b_u, linear_form_b_u)
solver.solve(b_u, u.vector)
# adjust u
c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
u.x.array[:] = u.x.array + c

k = 0
while (k < 1e2):
    cm_cmp_per_iter.append(compare_CM(subdomain, phi_exact, phi))
    delta_deri_phi.x.array[:] = delta_deri_tau(phi.x.array, tau)

    # cost function
    loss = assemble_scalar(form_loss)
    loss_per_iter.append(loss)

    # get w from u
    with b_w.localForm() as loc_w:
        loc_w.set(0)
    assemble_vector(b_w, linear_form_b_w)
    solver.solve(b_w, w.vector)

    # compute partial derivative of p from w
    J_p0 = assemble_scalar(form_J_p0)
    J_p1 = assemble_scalar(form_J_p1)
    J_p2 = assemble_scalar(form_J_p2)
    # condition: J_p
    print("k:", k)
    print("J_p0:", J_p0)
    print("J_p1:", J_p1)
    print("J_p2:", J_p2)
    if (abs(J_p0) < 1e-1 and abs(J_p1) < 1e-1 and abs(J_p2) < 1e-1):
        break
    # updata p from partial derivative
    # origin value
    p0v = p0.value.item()
    p1v = p1.value.item()
    p2v = p2.value.item()
    alpha = 1
    gamma = 0.3
    c = 0.2
    print("start line search")
    while (True):
        # adjust p
        p0.value = p0v - alpha * J_p0
        p1.value = p1v - alpha * J_p1
        p2.value = p2v - alpha * J_p2
        # compute u
        phi.interpolate(phi_model(p0, p1, p2))
        # phi.x.array[:] = (subdomain.geometry.x[:,0]-p0)**2 + (subdomain.geometry.x[:,1]-p1)**2 - p2**2
        delta_phi.x.array[:] = delta_tau(phi.x.array, tau)
        with b_u.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b_u, linear_form_b_u)
        solver.solve(b_u, u.vector)
        c = assemble_scalar(form_c1)/assemble_scalar(form_c2)
        u.x.array[:] = u.x.array + c
        # compute loss
        J = assemble_scalar(form_loss)
        J_cmp = J - (loss - c * alpha * (J_p0 ** 2 + J_p1 ** 2 + J_p2 ** 2))
        if (J_cmp < 0 or alpha < 1e-4):
            break
        alpha = gamma * alpha
    print("p0:", p0.value)
    print("p1:", p1.value)
    print("p2:", p2.value)
    k = k + 1
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

marker_exact = np.zeros(sub_node_num)
marker_exact[phi_exact.x.array < 0] = 1
marker0 = np.zeros(sub_node_num)
marker0[phi_0.x.array < 0] = 1
marker_result = np.zeros(sub_node_num)
marker_result[phi.x.array < 0] = 1

grid0 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid0.point_data["marker_exact"] = marker_exact
grid0.set_active_scalars("marker_exact")
grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid1.point_data["marker0"] = marker0
grid1.set_active_scalars("marker0")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid2.point_data["marker_result"] = marker_result
grid2.set_active_scalars("marker_result")

grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid3.point_data["phi_exact"] = phi_exact.x.array
grid3.set_active_scalars("phi_exact")
grid4 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid4.point_data["phi_0"] = phi_0.x.array
grid4.set_active_scalars("phi_0")
grid5 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid5.point_data["phi_result"] = phi.x.array
grid5.set_active_scalars("phi_result")

grid = (grid0, grid1, grid2, grid3, grid4, grid5)

plotter = pyvista.Plotter(shape=(2, 3))
for i in range(2):
    for j in range(3):
        plotter.subplot(i, j)
        plotter.add_mesh(grid[i*3 + j], show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
plotter.show()