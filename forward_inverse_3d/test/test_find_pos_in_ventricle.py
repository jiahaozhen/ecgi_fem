from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
from mpi4py import MPI

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("3d/heart_torso.msh", MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

array = subdomain.geometry.x
target_row = np.array([47.3, 40.9, 42.8])

distances = np.linalg.norm(array - target_row, axis=1)
nearest_indices = np.argsort(distances)[:6]
print(nearest_indices)