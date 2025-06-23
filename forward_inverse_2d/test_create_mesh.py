import gmsh
from mpi4py import MPI
import numpy as np

gmsh.initialize()
proc = MPI.COMM_WORLD.rank
top_marker = 2
bottom_marker = 1
left_marker = 1
if proc == 0:
    # We create one rectangle for each subdomain
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 0.5, tag=1)
    gmsh.model.occ.addRectangle(0, 0.5, 0, 1, 0.5, tag=2)
    # We fuse the two rectangles and keep the interface between them
    gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
    gmsh.model.occ.synchronize()

    # Mark the top (2) and bottom (1) rectangle
    top, bottom = None, None
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0.5, 0.25, 0]):
            bottom = surface[1]
        else:
            top = surface[1]
    gmsh.model.addPhysicalGroup(2, [bottom], bottom_marker)
    gmsh.model.addPhysicalGroup(2, [top], top_marker)
    # Tag the left boundary
    left = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0):
            left.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, left_marker)
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
gmsh.finalize()