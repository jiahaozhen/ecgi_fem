import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
import numpy as np

sys.path.append('.')
from utils.ventricular_segmentation_tools import get_ring_pts, distinguish_ring_pts, separate_lv_rv, get_apex_from_annulus_pts, lv_17_segmentation
from utils.visualize_tools import visualize_bullseye_points, visualize_bullseye_segment

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ring_pts_index, ring_pts = get_ring_pts(mesh_file, gdim)

mitral_ring, tricuspid_ring = distinguish_ring_pts(ring_pts)

lv_points, rv_points, lv_mask, rv_mask = separate_lv_rv(subdomain_ventricle.geometry.x, mitral_ring, tricuspid_ring)

apex = get_apex_from_annulus_pts(lv_points, mitral_ring)

seg_ids, r_mapped, theta_mapped = lv_17_segmentation(lv_points, mitral_ring, apex)

visualize_bullseye_segment(np.arange(17))

marker = np.zeros(subdomain_ventricle.geometry.x.shape[0], dtype=np.int32)
marker[lv_mask] = seg_ids
marker[rv_mask] = -1

plotter = pyvista.Plotter(off_screen=True)  # off_screen=True 防止弹出窗口
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data["f"] = marker
grid.set_active_scalars("f")

plotter.add_mesh(grid, show_edges=True)
plotter.view_yz()
plotter.add_axes()

# 保存图像
plotter.screenshot("ventricle_segments.png")  # 指定输出文件名
plotter.close()