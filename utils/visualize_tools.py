import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
from dolfinx.plot import vtk_mesh
import pyvista

from .function_tools import eval_function

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

def plot_standard_12_lead(standard12Lead, step_per_timeframe=4):
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    leads = [
        "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
        "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
    ]

    time = np.arange(0, standard12Lead.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
    for i, ax in enumerate(axs.flat):
        ax.plot(time, standard12Lead[:, i])
        ax.set_title(leads[i])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Potential (mV)')
        ax.grid(True)

    fig.suptitle('12-lead ECG', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_bsp_on_standard12lead(bsp_data, lead_index=np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1, step_per_timeframe=4):
    from .helper_function import transfer_bsp_to_standard12lead

    standard12Lead = transfer_bsp_to_standard12lead(bsp_data, lead_index)
    plot_standard_12_lead(standard12Lead, step_per_timeframe=step_per_timeframe)

def plot_v_random(v_data, step_per_timeframe=4):
    num_v = v_data.shape[1]  # 获取 v 的数量
    indices = np.random.choice(num_v, size=9, replace=False)
    # indices = np.where(v_data[450*step_per_timeframe, :] > -89)[0]  # 只选择有激活的点

    plt.figure(figsize=(12, 8))
    time = np.arange(0, v_data.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
    for i, idx in enumerate(indices):
        plt.plot(time, v_data[:, idx], label=f'v {idx}')
    plt.title("Randomly Selected V Data")
    plt.xlabel("Time")
    plt.ylabel("V Values")
    plt.legend()
    plt.grid(True)
    plt.show()