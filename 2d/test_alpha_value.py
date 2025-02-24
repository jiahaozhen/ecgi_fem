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
from helper_function import G_tau, delta_tau, delta_deri_tau, OuterBoundary1
import matplotlib.pyplot as plt

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))
# parameter a2 a1 a0 tau
a1 = -85
a2 = -25
a3 = 15
tau = 0.3
alpha = 1e-2
# phi G_phi delta_phi delta_deri_phi
phi_1 = Function(V2)
phi_2 = Function(V2)
phi_2_prior = Function(V2)
G_phi_1 = Function(V2)
G_phi_2 = Function(V2)
delta_phi_1 = Function(V2)
delta_phi_2 = Function(V2)
delta_deri_phi_1 = Function(V2)
delta_deri_phi_2 = Function(V2)
u = Function(V1)
w = Function(V1)
# function d
d = Function(V1)
# define d's value on the boundary
d_all_time = np.load(file='bsp_all_time.npy')
time_total = np.shape(d_all_time)[0]

# outerBoundary
facets = locate_entities_boundary(domain, tdim - 1, OuterBoundary1)
facet_indices = np.array(facets, dtype=np.int32)
facet_values = np.full(len(facet_indices), 1, dtype=np.int32)
facet_tags = meshtags(domain, tdim - 1, facet_indices, facet_values)
ds_out = Measure('ds', domain=domain, subdomain_data=facet_tags, subdomain_id=1)
dx2 = Measure("dx", domain=subdomain)

# function d
d = Function(V1)
loss_element_1 = 0.5 * (u - d) ** 2 * ds_out
loss_element_2 = 0.5 * alpha * (phi_2 - phi_2_prior) ** 2 * dx2 +\
    0.5 * alpha * dot(grad(phi_2 - phi_2_prior), grad(phi_2 - phi_2_prior)) * dx2
form_loss_1 = form(loss_element_1)
form_loss_2 = form(loss_element_2)

phi_2_all_time = np.load('phi_2_all_time.npy')

phi_2_prior.x.array[:] = np.ones(sub_node_num) * tau/2
loss_per_iter = np.zeros(time_total)
for i in range(time_total):
    phi_2.x.array[:] = phi_2_all_time[i]
    loss = assemble_scalar(form_loss_2)
    loss_per_iter[i] = loss
    phi_2_prior.x.array[:] = phi_2_all_time[i]

print(loss_per_iter)