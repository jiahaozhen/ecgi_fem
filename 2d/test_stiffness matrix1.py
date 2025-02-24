from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.mesh import create_submesh, locate_entities
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI

domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)

tdim = domain.topology.dim
subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(domain.topology.dim-2).size_local
num_cell = domain.topology.index_map(domain.topology.dim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))
V3 = functionspace(domain, ("DG", 0, (domain.geometry.dim, domain.geometry.dim)))

M = Function(V3)
matrix_dof = domain.geometry.dim*domain.geometry.dim
for i in range(matrix_dof):
    M.x.array[cell_markers.find(1)*matrix_dof + i] = 1
    M.x.array[cell_markers.find(2)*matrix_dof + i] = 2

# tree = bb_tree(domain, tdim)
# cell_candidates = compute_collisions_points(tree, points)
# colliding_cells = compute_colliding_cells(domain, cell_candidates, points)

Mi = Constant(domain, default_scalar_type(np.eye(tdim)))

u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx",domain=domain)
a1 = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form1 = form(a1)
A1 = assemble_matrix(bilinear_form1)
A1.assemble()

u2 = TestFunction(V2)
v2 = TrialFunction(V2)
dx2 = Measure("dx",domain=subdomain)
a2 = dot(grad(u2), dot(Mi, grad(v2)))*dx2
bilinear_form2 = form(a2)
A2 = assemble_matrix(bilinear_form2)
A2.assemble()