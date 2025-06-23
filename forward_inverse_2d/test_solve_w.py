from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure, ds
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
from helper_function import G_tau, delta_tau, delta_tau_s, petsc2array

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))
V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart 
#      M0 in Torso
M = Function(V3)
matrix_dof = tdim * tdim
for i in range(matrix_dof):
    if i in (1, 4, 9):
        M.x.array[cell_markers.find(1)*matrix_dof + i] = 1
        M.x.array[cell_markers.find(2)*matrix_dof + i] = 2
Mi = Constant(domain, default_scalar_type(np.eye(tdim)))

# parameter a2 a1 a0 tau
a2 = -90
a1 = -50
tau = 10

# matrix A_u
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx",domain=domain)
a = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form_a = form(a)
A_u = assemble_matrix(bilinear_form_a)
A_u.assemble()
A_u = petsc2array(A_u)
#matrix A_w
A_w = A_u.T

# find outer boundary
# define d on the outer boundary
# define integration on the outer boundary
d = Function(V1)

# vector B_w
u = Function(V1)
u3 = TestFunction(V1)
b_w = u3 * (u - d) * ds
linear_form_b_w = form(b_w)
B_w = assemble_vector(linear_form_b_w)