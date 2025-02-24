import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from ufl import grad, div, Measure, TestFunction, TrialFunction
from scipy.sparse import csr_matrix

sys.path.append('.')
from utils.helper_function import petsc2array

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("3d/data/mesh_multi_conduct_ecgsim.msh", MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim - 3).size_local

A = PETSc.Mat().create()
A.setSizes([sub_node_num, sub_node_num])
A.setType(PETSc.Mat.Type.MPIAIJ) 
A.setUp()

subdomain.topology.create_connectivity(1, 0)
line2point = subdomain.topology.connectivity(1, 0)

for l in range(line2point.num_nodes):
    idx = line2point.links(l)
    A.setValue(idx[0], idx[1], 1)
    A.setValue(idx[1], idx[0], 1)

A.assemble()
adj_matrix_np = petsc2array(A)
adj_matrix_np = csr_matrix(adj_matrix_np)
degrees = np.sum(adj_matrix_np, axis=1)

L = -A
for i in range(sub_node_num):
    L.setValue(i, i, degrees[i])
L.assemble()

LT = L.copy().transpose()
LT.assemble()

L_matrix = petsc2array(L)
LT_matrix = petsc2array(LT)
LTL_matrix = LT_matrix @ L_matrix

np.save('3d/data/LTL_matrix.npy', LTL_matrix)

V = functionspace(subdomain, ("Lagrange", 1))
u = TestFunction(V)
v = TrialFunction(V)
dx = Measure('dx', subdomain)
a_element = div(grad(u)) * div(grad(v)) * dx
form_a_element = form(a_element)
LTL_integral = assemble_matrix(form_a_element)
LTL_integral.assemble()
# a_element = div(grad(u)) * dx
# form_a_element = form(a_element)
# LTL_integral = assemble_vector(form_a_element)
# LTL_integral.view()
LTL_integral = petsc2array(LTL_integral)
np.save('3d/data/LTL_integral.npy', LTL_integral)

index_1 = np.nonzero(LTL_matrix)
index_2 = np.nonzero(LTL_integral)

print(index_1)
print(index_2)