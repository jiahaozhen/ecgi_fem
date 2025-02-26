import gmsh

gmsh.initialize()
gmsh.model.occ.addDisk(0,0,0,10,10,1) # Torso and Heart
gmsh.model.occ.addDisk(4,4,0,2,2,2)

gmsh.model.occ.cut([(2, 1)], [(2, 2)], 3, removeTool = False)
gmsh.model.occ.fragment([(2, 2)], [(2, 3)])
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [3], 1)
gmsh.model.addPhysicalGroup(2, [2], 2)

# gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
torso_entities = gmsh.model.getEntitiesForPhysicalGroup(2, 1) 
heart_entities = gmsh.model.getEntitiesForPhysicalGroup(2, 2) 
for entity in heart_entities:
    gmsh.model.mesh.setSize([(0, entity)], 0.1)
for entity in torso_entities:
    gmsh.model.mesh.setSize([(0, entity)], 1)

# gmsh.model.mesh.field.add("Box", 1)
# gmsh.model.mesh.field.setNumber(1, "VIn", 0.1)
# gmsh.model.mesh.field.setNumber(1, "VOut", 1)
# gmsh.model.mesh.field.setNumber(1, "XMin", 2)
# gmsh.model.mesh.field.setNumber(1, "XMax", 6)
# gmsh.model.mesh.field.setNumber(1, "YMin", 2)
# gmsh.model.mesh.field.setNumber(1, "YMax", 6)
# gmsh.model.mesh.field.setAsBackgroundMesh(1)

gmsh.model.mesh.generate(2)
gmsh.write("2d/data/heart_torso.msh")
gmsh.fltk.run()
gmsh.finalize()