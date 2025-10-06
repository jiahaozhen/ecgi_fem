import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
import numpy as np

sys.path.append('.')
from utils.helper_function import get_ring_pts, distinguish_ring_pts, separate_lv_rv, get_apex_from_annulus_pts, find_vertex_with_coordinate


def compute_lv_axis(annulus_points, apex_point):
    annulus_center = annulus_points.mean(axis=0)
    axis = apex_point - annulus_center
    axis /= np.linalg.norm(axis)
    return annulus_center, axis

def compute_lv_height_values(points, annulus_center, axis):
    return np.dot(points - annulus_center, axis)

def assign_segment(hi, theta_deg):
    if hi < 0.33:
        ring = 'basal'
        n_seg, offset = 6, 0
    elif hi < 0.66:
        ring = 'mid'
        n_seg, offset = 6, 6
    elif hi < 0.9:
        ring = 'apical'
        n_seg, offset = 4, 12
    else:
        return 16  # segment 17 (apex cap)

    theta_deg = (theta_deg + 360) % 360
    seg = int(theta_deg / (360 / n_seg)) + 1 + offset
    return seg

def lv_17_segmentation(lv_points, annulus_points, apex_point):
    annulus_center, axis = compute_lv_axis(annulus_points, apex_point)

    # 基于PCA确定环平面的基向量
    v1 = annulus_points[0] - annulus_center
    v1 -= np.dot(v1, axis) * axis
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)

    h_values = compute_lv_height_values(lv_points, annulus_center, axis)
    h_min, h_max = h_values.min(), h_values.max()
    h_norm = (h_values - h_min) / (h_max - h_min)

    seg_ids = []
    for p, h in zip(lv_points, h_norm):
        v = p - annulus_center
        v -= np.dot(v, axis) * axis
        theta = np.degrees(np.arctan2(np.dot(v, v2), np.dot(v, v1)))
        seg_ids.append(assign_segment(h, theta))
    return np.array(seg_ids)

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ring_pts_index, ring_pts = get_ring_pts(mesh_file, gdim)

left_ring_pts, right_ring_pts = distinguish_ring_pts(ring_pts)

lv_points, rv_points, lv_mask, rv_mask = separate_lv_rv(subdomain_ventricle.geometry.x, left_ring_pts, right_ring_pts)

apex = get_apex_from_annulus_pts(lv_points, left_ring_pts)

seg_id = lv_17_segmentation(lv_points, left_ring_pts, apex)

marker = np.zeros(subdomain_ventricle.geometry.x.shape[0], dtype=np.int32)
marker[lv_mask] = seg_id
marker[rv_mask] = -1

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data["f"] = marker
grid.set_active_scalars("f")
plotter.add_mesh(grid, show_edges=True)
plotter.view_yz()
plotter.add_axes()
plotter.show()