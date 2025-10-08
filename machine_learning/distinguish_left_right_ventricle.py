import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import numpy as np
import pyvista

sys.path.append('.')
from utils.ventricular_segmentation_tools import get_ring_pts, distinguish_ring_pts

def separate_lv_rv(points, mitral_ring, tricuspid_ring, offset_ratio=0.25):
    """
    points: Nx3 心室点云
    mitral_ring: Mx3 二尖瓣环
    tricuspid_ring: Kx3 三尖瓣环
    offset_ratio: 分界平面向右心室方向偏移比例 (0~0.5)
                  数值越大 → 左心室范围越大
    """
    # 1. 环中心
    cL = mitral_ring.mean(axis=0)
    cR = tricuspid_ring.mean(axis=0)
    
    # 2. 连线方向单位向量
    v = cR - cL
    dist = np.linalg.norm(v)
    v /= dist
    
    # 3. 分界平面中心（向右偏移）
    cM = (cL + cR) / 2
    cM_shift = cM + offset_ratio * dist * v  # 向右心室方向平移
    
    # 4. 点投影与分类
    proj = np.dot(points - cM_shift, v)
    lv_mask = proj < 0
    rv_mask = ~lv_mask

    return points[lv_mask], points[rv_mask]


mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ring_pts_index, ring_pts = get_ring_pts(mesh_file, gdim)

left_ring_pts, right_ring_pts = distinguish_ring_pts(ring_pts)

lv_points, rv_points = separate_lv_rv(subdomain_ventricle.geometry.x, left_ring_pts, right_ring_pts)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
plotter.add_mesh(grid, show_edges=True)

plotter.add_points(lv_points, color='red', point_size=10, render_points_as_spheres=True)
plotter.add_points(rv_points, color='green', point_size=10, render_points_as_spheres=True)

plotter.view_yz()
plotter.add_axes()
plotter.show()
