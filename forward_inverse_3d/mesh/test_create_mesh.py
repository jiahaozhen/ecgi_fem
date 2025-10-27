import gmsh

gmsh.initialize()
gmsh.model.occ.addSphere(0, 0, 0, 0.5, 1) # Torso and Heart
gmsh.model.occ.addSphere(0, 0, 0, 0.1, 2)

gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3, removeTool=False)
gmsh.model.occ.fragment([(3, 3)], [(3, 2)])
gmsh.model.occ.synchronize()

torso_points = gmsh.model.getBoundary([(3, 3)], recursive=True)
heart_points = gmsh.model.getBoundary([(3, 2)], recursive=True)
gmsh.model.mesh.setSize(torso_points, 0.1)
gmsh.model.mesh.setSize(heart_points, 0.02)

gmsh.model.addPhysicalGroup(3, [3], 1)
gmsh.model.addPhysicalGroup(3, [2], 2)

# gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)
gmsh.model.mesh.generate(3)
gmsh.fltk.run()
gmsh.write(r"forward_inverse_3d/data/heart_torso.msh")
gmsh.finalize()