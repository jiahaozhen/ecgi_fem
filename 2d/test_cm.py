from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary, meshtags
import numpy as np
from mpi4py import MPI

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local

coordinates = subdomain.geometry.x

V2 = functionspace(subdomain, ("Lagrange", 1))
phi_result = Function(V2)
phi_result.x.array[:] = np.full(phi_result.x.array.shape, -1)
phi_exact = np.load(file='forward_answer.npy')

marker_exact = np.full(phi_exact.shape, 0)
marker_exact[phi_exact < 0] = 1
marker_result = np.full(phi_result.x.array.shape, 0)
marker_result[phi_result.x.array < 0] = 1

coordinates_ischemic_exact = coordinates[np.where(marker_exact == 1)]
coordinates_ischemic_result = coordinates[np.where(marker_result == 1)]

cm1 = np.mean(coordinates_ischemic_exact, axis=0)
cm2 = np.mean(coordinates_ischemic_result, axis=0)

cm_compare = np.linalg.norm(cm1-cm2)