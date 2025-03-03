# compute potential u in B from transmembrane potential v
from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import pyvista

import sys
sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, eval_function

# mesh of Body
file = "2d/data/heart_torso.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local

num_node = domain.topology.index_map(tdim-2).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart 
#      M0 in Torso

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
V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
Mi = Constant(subdomain, default_scalar_type(np.eye(tdim)*sigma_i))

# phi_1 exact
class phi1_exact_solution():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2
        dist = np.sqrt(dist)
        return dist - self.r

# phi_2 exact
class phi2_exact_solution():
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2
        dist = np.sqrt(dist)
        return dist - self.t/20

# preparation 
a1 = -90 # no active no ischemia
a2 = -60 # no active ischemia
a3 = 10 # active no ischemia
a4 = -20 # active ischemia
tau = 0.05
time_total = 41

phi_1_exact = phi1_exact_solution(4, 6, 2)
phi_2_exact = phi2_exact_solution(4, 4, 0)
phi_1 = Function(V2)
phi_1.interpolate(phi_1_exact)
phi_2 = Function(V2)

delta_phi_1 = Function(V2)
delta_phi_2 = Function(V2)
delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)

G_phi_1 = Function(V2)
G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
G_phi_2 = Function(V2)

# A_u u = b_u
# matrix A
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx",domain=domain)
a = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form_a = form(a)
A = assemble_matrix(bilinear_form_a)
A.assemble()

solver = PETSc.KSP().create()
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# vector b_u
dx2 = Measure("dx",domain=subdomain)
# v = Function(V2)
# b_u_element = -dot(grad(u1), dot(Mi, grad(v))) * dx2
b_u_element = -(a1 - a2 - a3 + a4) * delta_phi_1 * G_phi_2 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 +\
        -(a1 - a2 - a3 + a4) * delta_phi_2 * G_phi_1 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2 +\
        -(a3 - a4) * delta_phi_1 * dot(grad(u1), dot(Mi, grad(phi_1))) * dx2 +\
        -(a2 - a4) * delta_phi_2 * dot(grad(u1), dot(Mi, grad(phi_2))) * dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b_u = form(b_u_element, entity_maps=entity_map)
b_u = create_vector(linear_form_b_u)

u = Function(V1)
u_all_time = np.zeros((time_total, num_node))
phi_2_all_time = np.zeros((time_total, sub_node_num))
v_exact_all_time = np.zeros((time_total, sub_node_num))

for t in range(time_total):
    phi_2_exact.t = t
    phi_2.interpolate(phi_2_exact)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
    # v.x.array[:] = (a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array +\
    #                 (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array)
    # v_exact_all_time[t] = v.x.array.copy()
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    phi_2_all_time[t] = phi_2.x.array.copy()
    u_all_time[t] = u.x.array.copy()
    v_exact_all_time[t] = (a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)) * G_phi_1.x.array +\
                        (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (1 - G_phi_1.x.array)

np.save('2d/data/bsp_all_time', u_all_time)
np.save('2d/data/phi_1', phi_1.x.array)
np.save('2d/data/phi_2_all_time', phi_2_all_time)
np.save('2d/data/v_exat_all_time.npy', v_exact_all_time)

v = Function(V2)
plotter = pyvista.Plotter(shape=(2, 5))
for i in range(2):
    for j in range(5):
        plotter.subplot(i, j)
        grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
        v.x.array[:] = v_exact_all_time[i*20 + j*5]
        grid.point_data["v"] = eval_function(v, subdomain.geometry.x)
        grid.set_active_scalars("v")
        # grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
        # phi_2.x.array[:] = phi_2_all_time[i*20 + j*5]
        # grid.point_data["phi_2"] = eval_function(phi_2, subdomain.geometry.x)
        # grid.set_active_scalars("phi_2")
        # grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
        # u.x.array[:] = u_all_time[i*20 + j*5]
        # grid.point_data["u"] = eval_function(u, domain.geometry.x)
        # grid.set_active_scalars("u")
        plotter.add_mesh(grid, show_edges=True)
        plotter.add_text(f"Time: {i*20 + j*5:.1f} ms")
        plotter.view_xy()
        plotter.add_axes()
plotter.show()