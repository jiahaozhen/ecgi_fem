import h5py
import gmsh
import numpy as np


# 只保留胸腔和球形心室
def create_mesh(source_file, target_file, lc, lc_ratio=4):
    """
    生成一个胸腔和球形心室的3D网格，心室用球体代替，球心和半径自动计算。
    Args:
        target_file (str): 网格输出文件路径
        lc (float): 网格特征长度
        lc_ratio (float): 心室区域网格细化比例
    """
    gmsh.initialize()
    gmsh.logger.start()
    gmsh.option.setNumber("General.Verbosity", 1)

    # 读取胸腔和心室表面
    mat = h5py.File(source_file, 'r')
    thorax_pts = np.array(mat['geom_thorax']['pts'])
    thorax_fac = np.array(mat['geom_thorax']['fac'], dtype=int) - 1
    ventricle_pts = np.array(mat['geom_ventricle']['pts'])

    # 球心设为原心室点的中心
    ventricle_pts = np.asarray(ventricle_pts)
    # Make sure ventricle_pts has shape (N, 3). If it's (3, N) transpose; if flat, reshape.
    if ventricle_pts.ndim == 1:
        ventricle_pts = ventricle_pts.reshape(-1, 3)
    elif ventricle_pts.shape[0] == 3 and ventricle_pts.shape[1] != 3:
        ventricle_pts = ventricle_pts.T
    ventricle_pts = ventricle_pts.reshape(-1, 3)
    center = ventricle_pts.mean(axis=0)
    # 半径为球心到胸腔点的最小距离的0.8倍，保证球体在胸腔内
    dists = np.linalg.norm(thorax_pts - center, axis=1)
    radius = np.min(dists) * 0.8

    # 创建胸腔
    def model_made_points(p, pindex):
        points = [gmsh.model.occ.addPoint(*pt, lc) for pt in p]
        lines = np.zeros([len(pindex), len(pindex[0])])
        edges = {}
        for i in range(len(pindex)):
            for j in range(len(pindex[i])):
                # check if the same edge appeared
                p1 = points[pindex[i][j - 1]]
                p2 = points[pindex[i][j]]
                if (p2, p1) in edges:
                    lines[i][j] = -edges[(p2, p1)]
                    continue
                edges[(p1, p2)] = gmsh.model.occ.addLine(p1, p2)
                lines[i][j] = edges[(p1, p2)]
        cloops = [gmsh.model.occ.addCurveLoop(lines[i]) for i in range(len(lines))]
        faces = [
            gmsh.model.occ.addPlaneSurface([cloops[i]]) for i in range(len(cloops))
        ]
        surface_loop = gmsh.model.occ.addSurfaceLoop(faces)
        volume = gmsh.model.occ.addVolume([surface_loop])
        return volume

    model_thorax = model_made_points(thorax_pts, thorax_fac)

    # 创建球形心室
    model_ventricle = gmsh.model.occ.addSphere(*center, radius)

    # 切割胸腔中的心室
    gmsh.model.occ.cut(
        [(3, model_thorax)], [(3, model_ventricle)], tag=7, removeTool=False
    )
    gmsh.model.occ.fragment([(3, 7)], [(3, model_ventricle)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [7], 1)  # Torso
    gmsh.model.addPhysicalGroup(3, [model_ventricle], 2)  # Ventricle

    # mesh size
    lc_ventricle = lc / lc_ratio
    lc_other = lc

    # 创建 Box 尺寸场，使球体心室区域更细
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", lc_ventricle)
    gmsh.model.mesh.field.setNumber(1, "VOut", lc_other)
    gmsh.model.mesh.field.setNumber(1, "XMin", center[0] - radius)
    gmsh.model.mesh.field.setNumber(1, "XMax", center[0] + radius)
    gmsh.model.mesh.field.setNumber(1, "YMin", center[1] - radius)
    gmsh.model.mesh.field.setNumber(1, "YMax", center[1] + radius)
    gmsh.model.mesh.field.setNumber(1, "ZMin", center[2] - radius)
    gmsh.model.mesh.field.setNumber(1, "ZMax", center[2] + radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(3)
    gmsh.write(target_file)
    gmsh.finalize()


if __name__ == '__main__':
    lc = 40
    source_file = r'forward_inverse_3d/data/raw_data/geom_normal_male.mat'
    target_file = r'forward_inverse_3d/data/mesh/mesh_normal_male[sphere].msh'
    create_mesh(source_file, target_file, lc, lc_ratio=4)
