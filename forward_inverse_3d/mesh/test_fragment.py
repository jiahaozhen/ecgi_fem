import gmsh
import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh, locate_entities_boundary
from mpi4py import MPI

file_fragment = "3d/data/heart_torso_fragment.msh"
file_nofragment = "3d/data/heart_torso_nofragment.msh"

def create_mesh(fragment_flag):
    gmsh.initialize()
    gmsh.model.occ.addSphere(0, 0, 0, 1, 1) # Torso and Heart
    gmsh.model.occ.addSphere(0, 0, 0, 0.5, 2)

    gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3, removeTool=False)
    if fragment_flag:
        gmsh.model.occ.fragment([(3, 3)], [(3, 2)])
        file = file_fragment
    else:
        file = file_nofragment

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [3], 1)
    gmsh.model.addPhysicalGroup(3, [2], 2)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
    gmsh.model.mesh.generate(3)
    # gmsh.fltk.run()
    gmsh.write(file)
    gmsh.finalize()

# create_mesh(True)
# create_mesh(False)

domain1, cell_markers1, _ = gmshio.read_from_msh('3d/data/mesh_multi_conduct_ecgsim_fragment.msh', MPI.COMM_WORLD, gdim=3)
tdim = domain1.topology.dim
subdomain1, _, _, _ = create_submesh(domain1, tdim, cell_markers1.find(2))

domain2, cell_markers2, _ = gmshio.read_from_msh('3d/data/mesh_multi_conduct_ecgsim_no_fragment.msh', MPI.COMM_WORLD, gdim=3)
subdomain2, _, _, _ = create_submesh(domain2, tdim, cell_markers2.find(2))

boundary1 = locate_entities_boundary(domain1, 0, lambda x: np.full(x.shape[1], True, dtype=bool))
boundary2 = locate_entities_boundary(domain2, 0, lambda x: np.full(x.shape[1], True, dtype=bool))
subboundary1 = locate_entities_boundary(subdomain1, 0, lambda x: np.full(x.shape[1], True, dtype=bool))
subboundary2 = locate_entities_boundary(subdomain2, 0, lambda x: np.full(x.shape[1], True, dtype=bool))

print(boundary1.shape)
print(boundary2.shape)
print(subboundary1.shape)
print(subboundary2.shape)