import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function
from mpi4py import MPI

# mesh of Body
file = "2d/data/heart_torso.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local

phi_2_all_time = np.load("2d/data/phi_2_all_time.npy")

# compute the norm of differenec between adjcent timeframe
phi_2_norm = np.zeros(phi_2_all_time.shape[0]-2)
for i in range(phi_2_all_time.shape[0]-2):
    phi_2_norm[i] = np.linalg.norm(phi_2_all_time[i+2]+phi_2_all_time[i]-2*phi_2_all_time[i+1])
    # phi_2_norm[i] = np.linalg.norm(phi_2_all_time[i+2] + phi_2_all_time[i] - 2 * phi_2_all_time[i+1])
    print("phi_2_norm[", i, "]: ", phi_2_norm[i])