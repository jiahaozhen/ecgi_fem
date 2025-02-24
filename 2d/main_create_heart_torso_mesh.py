import gmsh
from mpi4py import MPI


gmsh.initialize()
gmsh.model.occ.addDisk(0,0,0,10,10,1) # Torso and Heart
gmsh.model.occ.addDisk(4,4,0,2,2,2)
# gmsh.model.occ.addDisk(4,4,0,2,2,3)
# gmsh.model.occ.addDisk(5,4,0,0.5,0.5,4)
# gmsh.model.occ.addDisk(3,4,0,0.5,0.5,5)

# gmsh.model.occ.cut([(2, 1)], [(2, 2)], 6)
# gmsh.model.occ.cut([(2, 3)], [(2, 4), (2, 5)], 7)
# gmsh.model.occ.fragment([(2, 6)], [(2, 7)])
gmsh.model.occ.cut([(2, 1)], [(2, 2)], 3, removeTool = False)
gmsh.model.occ.fragment([(2, 2)], [(2, 3)])
gmsh.model.occ.synchronize()

# gmsh.model.addPhysicalGroup(2, [6], 1)  # Torso as physical group 1
# gmsh.model.addPhysicalGroup(2, [7], 2)  # Heart as physical group 2
gmsh.model.addPhysicalGroup(2, [3], 1)
gmsh.model.addPhysicalGroup(2, [2], 2)

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
gmsh.model.mesh.generate(2)
gmsh.write("2d/data/heart_torso.msh")
gmsh.fltk.run()
gmsh.finalize()