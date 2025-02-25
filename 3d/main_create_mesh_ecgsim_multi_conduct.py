import h5py
import gmsh
import numpy as np

def create_mesh(target_file, lc, multi_flag=True):
    """
    Generates a 3D mesh for anatomical structures from geometry data stored in a .mat file using the GMSH meshing library.

    Args:
        target_file (str): The path to the output file where the mesh will be saved.
        lc (float): The characteristic length used for mesh refinement.
        multi_flag (bool, optional): Flag indicating whether multiple anatomical structures should be meshed. 
                                      If True, includes lung and cavitary regions in the mesh. Defaults to True.

    Returns:
        None
    """
    geom_data = h5py.File('3d/data/geom_ecgsim.mat', 'r')

    geom_ventricle = geom_data['geom_ventricle']
    geom_thorax = geom_data['geom_thorax']
    geom_lcav = geom_data['geom_lcav']
    geom_rcav = geom_data['geom_rcav']
    geom_llung = geom_data['geom_llung']
    geom_rlung = geom_data['geom_rlung']

    geom_thorax_fac = np.array(geom_thorax['fac'], dtype = np.int32) - 1
    geom_thorax_pts = np.array(geom_thorax['pts'])
    geom_ventricle_fac = np.array(geom_ventricle['fac'], dtype = np.int32) - 1
    geom_ventricle_pts = np.array(geom_ventricle['pts'])
    geom_llung_fac = np.array(geom_llung['fac'], dtype = np.int32) - 1
    geom_llung_pts = np.array(geom_llung['pts'])
    geom_rlung_fac = np.array(geom_rlung['fac'], dtype = np.int32) - 1
    geom_rlung_pts = np.array(geom_rlung['pts'])
    geom_lcav_fac = np.array(geom_lcav['fac'], dtype = np.int32) - 1
    geom_lcav_pts = np.array(geom_lcav['pts'])
    geom_rcav_fac = np.array(geom_rcav['fac'], dtype = np.int32) - 1
    geom_rcav_pts = np.array(geom_rcav['pts'])

    def model_made_points(p, pindex):
        if lc == None:
            points = [gmsh.model.occ.addPoint(*pt) for pt in p]
        else:
            points = [gmsh.model.occ.addPoint(*pt, lc) for pt in p]
        lines = np.zeros([len(pindex), len(pindex[0])])
        edges = {}
        for i in range(len(pindex)):
            for j in range(len(pindex[i])):
                # check if the same edge appeared
                p1 = points[pindex[i][j-1]]
                p2 = points[pindex[i][j]]
                if (p2, p1) in edges:
                    lines[i][j] = -edges[(p2, p1)]
                    continue
                edges[(p1, p2)] = gmsh.model.occ.addLine(p1, p2)
                lines[i][j] = edges[(p1, p2)]
        cloops = [gmsh.model.occ.addCurveLoop(lines[i]) for i in range(len(lines))]
        faces = [gmsh.model.occ.addPlaneSurface([cloops[i]]) for i in range(len(cloops))]
        surface_loop = gmsh.model.occ.addSurfaceLoop(faces)
        volume = gmsh.model.occ.addVolume([surface_loop])
        return volume
    
    gmsh.initialize()
    gmsh.logger.start()
    gmsh.option.setNumber("General.Verbosity", 1)
    model_ventricle = model_made_points(geom_ventricle_pts, geom_ventricle_fac)
    model_thorax = model_made_points(geom_thorax_pts, geom_thorax_fac)
    if multi_flag == True:
        model_llung = model_made_points(geom_llung_pts, geom_llung_fac)
        model_rlung = model_made_points(geom_rlung_pts, geom_rlung_fac)
        model_lcav = model_made_points(geom_lcav_pts, geom_lcav_fac)
        model_rcav = model_made_points(geom_rcav_pts, geom_rcav_fac)

        gmsh.model.occ.cut([(3, model_thorax)], [(3, model_ventricle), (3, model_lcav), (3, model_rcav), 
                                                (3, model_llung), (3, model_rlung)], tag = 7, removeTool = False)
        gmsh.model.occ.fragment([(3, 7)], [(3, model_ventricle), 
                                        (3, model_lcav), (3, model_rcav), (3, model_llung), (3, model_rlung)])

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [7], 1)  # Torso as physical group 1
        gmsh.model.addPhysicalGroup(3, [model_ventricle], 2)  # Heart as physical group 2
        gmsh.model.addPhysicalGroup(3, [model_llung, model_rlung], 3) # Lung as physical group 3
        gmsh.model.addPhysicalGroup(3, [model_lcav, model_rcav], 4) # Cav as physical group 4
    else:
        gmsh.model.occ.cut([(3, model_thorax)], [(3, model_ventricle)], tag = 7, removeTool = False)
        gmsh.model.occ.fragment([(3, 7)], [(3, model_ventricle)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [7], 1)  # Torso as physical group 1
        gmsh.model.addPhysicalGroup(3, [model_ventricle], 2)  # Heart as physical group 2

    gmsh.model.mesh.generate(3)
    gmsh.write(target_file)
    # log = "\n".join(gmsh.logger.get())
    # with open('3d/gmsh_log.txt', 'w') as log_file:
    #     log_file.write("log:\n")
        # log_file.write(log)
    gmsh.finalize()

if __name__ == '__main__':
    lc = 10
    file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    create_mesh(file, lc)