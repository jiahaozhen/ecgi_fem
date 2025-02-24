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

Mi = Constant(domain, default_scalar_type(np.eye(tdim)))

def delta_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = (1 / (2 * tau)) * (1 + np.cos(np.pi * s[condition3] / tau))
    return result

V = functionspace(subdomain, ("Lagrange", 1))

a1 = -45
a2 = -85
tau = 0.1
phi = Function(V)
phi.vector[:] = np.ones(len(phi.vector.array))*a2
composite_f = Function(V)
composite_f.vector[:] = delta_tau(phi.vector.array, tau)

u = TestFunction(V)
v = TrialFunction(V)
dx2 = Measure("dx",domain=subdomain)
a = (a1-a2) * composite_f * dot(grad(u), dot(Mi, grad(v)))*dx2
bilinear_form2 = form(a)
A = assemble_matrix(bilinear_form2)
A.assemble()