# compute potential u in B from transmembrane potential v
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form
from dolfinx.fem.petsc import create_vector, assemble_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
from ufl import TestFunction, Measure
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pyvista

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

# print(sub_to_parent)

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

V1 = functionspace(domain, ("Lagrange", 1))

def OuterBoundary(x):
    return x[0]**2 + x[1]**2 > 64

outerBoundary = locate_entities_boundary(domain, domain.topology.dim-2, OuterBoundary)

print(outerBoundary)

u = TestFunction(V1)
ds = Measure("ds", domain=domain)
b_w = u * ds
entity_map = {domain._cpp_object: outerBoundary}
linear_form_b_w = form(b_w, entity_maps=entity_map)
B_w = assemble_vector(linear_form_b_w)

print(B_w.array)