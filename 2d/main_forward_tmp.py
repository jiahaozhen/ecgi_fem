# compute potential u in B from transmembrane potential v
from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure, FacetNormal
from mpi4py import MPI
from petsc4py import PETSc
import pyvista

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("2d/heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))
u = Function(V1)
v = Function(V2)
# print(subdomain.geometry.x[np.where(subdomain.geometry.x==47.3)[0]])
# print(subdomain.geometry.x)

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart
#      M0 in Torso 
# mS/mm
M0_value = 1
Mi_value = 1
Me_value = 1
def rho1(x):
    tensor = np.eye(tdim) * M0_value
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(tdim) * (Mi_value + Me_value)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
Mi = Constant(subdomain, default_scalar_type(np.eye(tdim) * Mi_value))

# A u = b
# matrix A
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx", domain = domain)
a_element = dot(grad(u1), dot(M, grad(v1))) * dx1
bilinear_form_a = form(a_element)
A = assemble_matrix(bilinear_form_a)
A.assemble()

# b
dx2 = Measure("dx", domain = subdomain)
b_element = -dot(grad(u1), dot(Mi, grad(v))) * dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b = form(b_element, entity_maps = entity_map)
b = create_vector(linear_form_b)

solver = PETSc.KSP().create()
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# v_value = np.full_like(v.x.array, -90)
# array = subdomain.geometry.x
# target_row = np.array([4, 4, 0])
# distances = np.linalg.norm(array - target_row, axis=1)
# nearest_indices = np.argsort(distances)[:40]
# v_value[nearest_indices] = -60
v.x.array[:] = np.random.random(subdomain.geometry.x.shape[0]) * 90
assemble_vector(b, linear_form_b)
solver.solve(b, u.vector)

np.save('2d/u.npy', u.x.array)
np.save('2d/v.npy', v.x.array)

# test boundary coditions
# u_t = u_e
nh = FacetNormal(subdomain)
nt = FacetNormal(domain)
ds1 = Measure('ds', domain)
ds2 = Measure('ds', subdomain)
# sigma_i * grad(u_e) * n_h = - sigma_i * grad(v) * n_h
bc1_e = dot(dot(Mi, grad(u + v)), nh) * ds2
form_bc1 = form(bc1_e, entity_maps = entity_map)
bc1 = assemble_scalar(form_bc1)
# sigma_t * grad(u_t) * n_h = sigma_e * grad(u_e) * n_h
bc2_e = (dot(dot(M, grad(u)), nh) - dot(dot(M, grad(u)), nh)) * ds2
form_bc2 = form(bc2_e, entity_maps = entity_map)
bc2 = assemble_scalar(form_bc2)
# sigma_t * grad(u_t) * n_t = 0
bc3_e = dot(dot(M, grad(u)), nt) * ds1
form_bc3 = form(bc3_e)
bc3 = assemble_scalar(form_bc3)
# print(bc1, bc2, bc3)

grid1 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid1.point_data["u"] = u.x.array
grid1.set_active_scalars("u")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid2.point_data["v"] = v.x.array
grid2.set_active_scalars("v")
grid = (grid1, grid2)

plotter = pyvista.Plotter(shape=(1, 2))
for i in range(2):
    plotter.subplot(0, i)
    plotter.add_mesh(grid[i], show_edges=True)
    plotter.view_xy()
    plotter.add_axes()
plotter.show()