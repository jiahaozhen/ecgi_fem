import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from sklearn.decomposition import PCA
import numpy as np
import pyvista
import matplotlib.pyplot as plt

def project_point_on_line(point, direction_vector):
    """
    将一个点投影到过原点且给定方向的直线上。

    Args:
        point (np.array): 表示点的 NumPy 数组 (e.g., [x, y, z])。
        direction_vector (np.array): 表示直线方向的 NumPy 数组 (e.g., [dx, dy, dz])。

    Returns:
        np.array: 投影点的坐标。
    """
    direction_vector = np.asarray(direction_vector) # 确保是 NumPy 数组
    point = np.asarray(point) # 确保是 NumPy 数组

    # 计算投影分量：(a . b) / ||b||^2
    projection_scalar = np.dot(point, direction_vector) / np.dot(direction_vector, direction_vector)

    # 投影点 = 投影分量 * 方向向量
    projected_point = projection_scalar * direction_vector
    return projected_point

def find_farthest_projected_point(points, direction_vector):
    """
    找到给定点集中，在投影到指定直线后，距离原点最远的投影点。

    Args:
        points (list of np.array): 包含多个点的列表。
        direction_vector (np.array): 表示直线方向的 NumPy 数组。

    Returns:
        tuple: (最远投影点, 原始点)
    """
    farthest_point = None
    original_point_of_farthest = None
    max_distance_sq = -1 # 使用平方距离避免开方，提高效率

    for point in points:
        projected_point = project_point_on_line(point, direction_vector)
        # 计算投影点到原点的平方距离
        current_distance_sq = np.dot(projected_point, projected_point)

        if current_distance_sq > max_distance_sq:
            max_distance_sq = current_distance_sq
            farthest_point = projected_point
            original_point_of_farthest = point
    return farthest_point, original_point_of_farthest

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

sample_points = subdomain_ventricle.geometry.x
line_point_0 = np.array([0, 0, 0])  # 直线上的参考点
line_direction_vector = np.array([1.0, 1.0, -1.0])  # 直线的方向向量
line_t = np.linspace(-5, 5, 100) # 参数 t 的范围
line_x = line_point_0[0] + line_t * line_direction_vector[0]
line_y = line_point_0[1] + line_t * line_direction_vector[1]
line_z = line_point_0[2] + line_t * line_direction_vector[2]

farthest_proj_point, original_point = find_farthest_projected_point(sample_points, line_direction_vector)
if farthest_proj_point is not None:
    print(f"在所有投影点中，距离原点最远的投影点是: {farthest_proj_point}")
    print(f"该投影点对应的原始点是: {original_point}")
    print(f"该投影点到原点的距离是: {np.linalg.norm(farthest_proj_point):.4f}")
else:
    print("没有可用的点进行投影。")
tip_point = original_point
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
plotter.add_mesh(grid, show_edges=True)
plotter.add_points(np.array([tip_point]), color='red', point_size=10, render_points_as_spheres=True)
plotter.view_yz()
plotter.add_axes()
plotter.show()
