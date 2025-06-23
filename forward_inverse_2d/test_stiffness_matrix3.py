from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form
from dolfinx.mesh import create_submesh, meshtags
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
import numpy as np
# function from different space integration on submesh
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)

tdim = domain.topology.dim
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

entity_map = meshtags(subdomain, tdim, np.arange(len(sub_to_parent), dtype=np.int32), sub_to_parent)

# print(entity_map)

num_node = domain.topology.index_map(domain.topology.dim-2).size_local
num_cell = domain.topology.index_map(domain.topology.dim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

entity_map = {domain._cpp_object: sub_to_parent}

u = TestFunction(V1)
v = TrialFunction(V2)
dx = Measure("dx", subdomain)
# print(dx.integral_type)
a = dot(grad(u), grad(v))*dx

bilinear_form = form(a, entity_maps=entity_map)
A = assemble_matrix(bilinear_form)
A.assemble()
print(A.getSize())