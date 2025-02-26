import h5py
import gmsh
import numpy as np

def create_mesh(target_file, lc):
    geom_data = h5py.File('3d/data/geom_ecgsim.mat', 'r')

    geom_ventricle = geom_data['geom_ventricle']
    geom_ventricle_fac = np.array(geom_ventricle['fac'], dtype = np.int32) - 1
    geom_ventricle_pts = np.array(geom_ventricle['pts'])

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
    gmsh.option.setNumber("General.Verbosity", 3)
    model_ventricle = model_made_points(geom_ventricle_pts, geom_ventricle_fac)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [model_ventricle], 1)
    gmsh.model.mesh.generate(3)
    gmsh.write(target_file)
    gmsh.finalize()

if __name__ == '__main__':
    lc = 10
    target_file = '3d/data/mesh_ecgsim_ventricle.msh'
    create_mesh(target_file, lc)