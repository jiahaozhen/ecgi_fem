import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
from dolfinx.plot import vtk_mesh
import pyvista

from .helper_function import eval_function

def visualize_bullseye_points(theta, r, val):
    """在 bullseye 坐标上绘制点云分布"""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    ax.set_ylim(0, 1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    sc = ax.scatter(np.radians(theta), r, c=val, cmap='tab20', s=5)
    plt.title("LV Points mapped to AHA Bullseye", fontsize=14)
    plt.colorbar(sc, label="Segment ID")
    plt.show()

def visualize_bullseye_segment(values):
    """
    可视化 AHA 17 段 bullseye 图
    values: 长度为17的数组, 对应17个分区的值
    """
    assert len(values) == 17, "需要17个分区的值"

    # 17段映射
    # basal: 6段, mid: 6段, apical: 4段, apex: 1段
    segments = [
        {'start': 0, 'end': 6, 'radius': [0.66, 1.0]},   # basal
        {'start': 6, 'end': 12, 'radius': [0.33, 0.66]}, # mid
        {'start': 12, 'end': 16, 'radius': [0.1, 0.33]}, # apical
        {'start': 16, 'end': 17, 'radius': [0.0, 0.1]}, # apex中心点
    ]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)

    cmap = get_cmap('viridis')
    norm = Normalize(vmin=min(values), vmax=max(values))

    for seg in segments:
        start, end = seg['start'], seg['end']
        r_inner, r_outer = seg['radius']
        n = end - start
        for i in range(n):
            theta_start = 2 * np.pi * i / n
            theta_end = 2 * np.pi * (i + 1) / n
            color = cmap(norm(values[start + i]))
            ax.bar(
                x=(theta_start + theta_end)/2, 
                height=r_outer-r_inner, 
                width=(theta_end - theta_start), 
                bottom=r_inner, 
                color=color,
                edgecolor='white',
                align='center'
            )

    plt.show()

def plot_val_on_mesh(mesh_file, val, gdim=3, target_cell=None, name="f", title="Function on Mesh", f_val_flag=False):
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.mesh import create_submesh
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain
    if f_val_flag:
        from dolfinx.fem import Function, functionspace
        f = Function(functionspace(domain, ("Lagrange", 1)))
        f.x.array[:] = val
        plot_f_on_domain(domain, f, name=name, tdim=domain.topology.dim, title=title)
    else:
        plot_val_on_domain(domain, val, name=name, tdim=domain.topology.dim, title=title)

def plot_f_on_domain(domain, f, name="f", tdim=3, title="Function on Domain"):
    val = eval_function(f, domain.geometry.x)
    plot_val_on_domain(domain, val, name=name, tdim=tdim, title=title)

def plot_val_on_domain(domain, val, name="val", tdim=3, title="Value on Domain"):
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    grid.point_data[name] = val
    grid.set_active_scalars(name)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_title(title)
    plotter.view_yz()
    plotter.add_axes()
    plotter.show(auto_close=False)