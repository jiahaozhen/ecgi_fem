from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
import h5py
import numpy as np
from mpi4py import MPI

tmp_file = h5py.File('3d/ecgsim_tmp.mat')
tmp_data = np.array(tmp_file['tmp'])
total_time = tmp_data.shape[0]
print(tmp_data.shape[0])

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("3d/heart_torso.msh", MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

array = subdomain.geometry.x
geom_data = h5py.File('3d/geom_ecgsim.mat', 'r')
geom_ventricle = geom_data['geom_ventricle']
geom_ventricle_pts = np.array(geom_ventricle['pts'])
indices = []
for row in geom_ventricle_pts:
    matches = np.all(array == row, axis = 1)
    indice = np.where(matches)[0]
    indices.append(indice.item())

np.save('3d/switch_indices.npy', indices)
print(indices)