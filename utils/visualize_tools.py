import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
import pyvista
from mpi4py import MPI
from dolfinx.mesh import create_submesh
from dolfinx.fem import Function, functionspace
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh

from .function_tools import eval_function
from .signal_processing_tools import transfer_bsp_to_standard12lead, smooth_ecg_mean

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
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain
    if f_val_flag:
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

def compare_bsp_on_standard12lead(*bsp_datas, 
                                 case_name='normal_male',
                                 labels=None, 
                                 step_per_timeframe=4,
                                 filter_flag=True,
                                 filter_window_size=50):

    standard12Leads = [transfer_bsp_to_standard12lead(bsp_data, case_name) for bsp_data in bsp_datas]
    compare_standard_12_lead(*standard12Leads,
                             labels=labels,
                             step_per_timeframe=step_per_timeframe,
                             filter_flag=filter_flag,
                             filter_window_size=filter_window_size)


def compare_standard_12_lead(*standard12Leads, 
                             labels=None, 
                             step_per_timeframe=4, 
                             filter_flag=True,
                             filter_window_size=50):
    if filter_flag:
        standard12Leads = [smooth_ecg_mean(data, window_size=filter_window_size) for data in standard12Leads]

    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    leads = [
        "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
        "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
    ]

    time = np.arange(0, standard12Leads[0].shape[0] / step_per_timeframe, 1 / step_per_timeframe)
    for i, ax in enumerate(axs.flat):
        for idx, data in enumerate(standard12Leads):
            label = labels[idx] if labels and idx < len(labels) else f"Data {idx + 1}"
            linestyle = '-' if idx == 0 else '--'
            ax.plot(time, data[:, i], linestyle=linestyle, label=label)
        ax.set_title(leads[i])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Potential (mV)')
        ax.legend()
        ax.grid(True)

    fig.suptitle('Comparison of 12-lead ECG', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_bsp_on_standard12lead(bsp_data, 
                               case_name='normal_male',
                               step_per_timeframe=4,
                               filter_flag=True,
                               filter_window_size=50):
    standard12Lead = transfer_bsp_to_standard12lead(bsp_data, case_name=case_name)
    plot_standard_12_lead(standard12Lead, 
                          step_per_timeframe=step_per_timeframe,
                          filter_flag=filter_flag,
                          filter_window_size=filter_window_size
                          )
    
def plot_standard_12_lead(standard12Lead, 
                          step_per_timeframe=4, 
                          filter_flag=True, 
                          filter_window_size=50):
    if filter_flag:
        standard12Lead = smooth_ecg_mean(standard12Lead, window_size=filter_window_size)
    
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    leads = [
        "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
        "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
    ]

    time = np.arange(0, standard12Lead.shape[0] / step_per_timeframe, 1 / step_per_timeframe)

    for idx, ax in enumerate(axs.flat):
        row = idx // 3
        col = idx % 3

        ax.plot(time, standard12Lead[:, idx])
        ax.set_title(leads[idx])

        # y label only at the first column
        if col == 0:
            ax.set_ylabel("Potential (mV)")
        else:
            ax.set_ylabel("")

        # x label only at the bottom row
        if row == 3:
            ax.set_xlabel("Time (ms)")
        else:
            ax.set_xlabel("")

        # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(True)

    fig.suptitle("12-lead ECG", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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

def plot_convergence(summary, base_lc=20, base_lc_ratio=1):
    """
    绘制网格收敛性图（多指标）
    """
    lc_vals = [f"{lc}-{lc_ratio}" for lc, lc_ratio in sorted(summary.keys())]
    corr_vals = [summary[key]['corr'] for key in sorted(summary.keys())]
    relL2_vals = [summary[key]['rel_L2'] for key in sorted(summary.keys())]
    rmse_vals = [summary[key]['rmse'] for key in sorted(summary.keys())]
    peak_shift_vals = [summary[key]['peak_shift'] for key in sorted(summary.keys())]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    axs[0].plot(lc_vals, corr_vals, 'o-', label='Correlation (r)')
    axs[0].set_title(f"Mean Correlation vs lc-lc_ratio (base lc={base_lc}, lc_ratio={base_lc_ratio})")
    axs[0].set_xlabel("lc-lc_ratio")
    axs[0].set_ylabel("Correlation (r)")
    axs[0].grid(True)

    axs[1].plot(lc_vals, relL2_vals, 's--', color='tab:red', label='Relative L2 Error')
    axs[1].set_title("Relative L2 Error vs lc-lc_ratio")
    axs[1].set_xlabel("lc-lc_ratio")
    axs[1].set_ylabel("Relative L2 Error")
    axs[1].grid(True)

    axs[2].plot(lc_vals, rmse_vals, 'd--', color='tab:orange', label='RMSE')
    axs[2].set_title("RMSE vs lc-lc_ratio")
    axs[2].set_xlabel("lc-lc_ratio")
    axs[2].set_ylabel("RMSE")
    axs[2].grid(True)

    axs[3].plot(lc_vals, peak_shift_vals, 'x-', color='tab:green', label='Peak Shift')
    axs[3].set_title("Peak Time Shift vs lc-lc_ratio")
    axs[3].set_xlabel("lc-lc_ratio")
    axs[3].set_ylabel("Δt_peak (samples)")
    axs[3].grid(True)

    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_triangle_mesh(points, triangles, point_values=None, cell_values=None, title=None):

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    # 2D → 自动扩展到 3D
    if points.shape[1] == 2:
        pts3d = np.c_[points, np.zeros(points.shape[0])]
    else:
        pts3d = points

    # PyVista cells 格式
    cells = np.hstack([np.c_[np.full(len(triangles), 3), triangles]]).ravel().astype(np.int64)
    celltypes = np.full(len(triangles), 5, dtype=np.uint8)  # VTK_TRIANGLE=5

    mesh = pyvista.UnstructuredGrid(cells, celltypes, pts3d)

    # -------- 在 mesh 之上赋值 --------
    if point_values is not None:
        mesh.point_data["point_val"] = np.asarray(point_values, dtype=float)

    if cell_values is not None:
        mesh.cell_data["cell_val"] = np.asarray(cell_values, dtype=float)

    # -------- 绘制 --------
    p = pyvista.Plotter()

    if point_values is not None:
        p.add_mesh(mesh, scalars="point_val", show_edges=True)
    elif cell_values is not None:
        p.add_mesh(mesh, scalars="cell_val", show_edges=True)
    else:
        p.add_mesh(mesh, show_edges=True, color="lightgray")

    if title is not None:
        p.add_title(title)

    p.add_axes()
    p.show()

def scatter_val_on_domain(domain, val, name="val", tdim=3, title="Value on Domain", activation_dict=None):
    pts = domain.geometry.x
    val = np.asarray(val)

    if len(val) != pts.shape[0]:
        raise ValueError(f"val 长度 {len(val)} 与点数 {pts.shape[0]} 不匹配")

    fig = plt.figure(figsize=(7, 5))

    # 使用 tdim 与实际点维度共同决定绘图是 2D 还是 3D（确保 tdim 被使用以消除未使用警告）
    effective_dim = min(tdim, pts.shape[1])

    # 如果是 3D mesh
    if effective_dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=val, s=4)
    else:
        # 2D mesh
        ax = fig.add_subplot(111)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=val, s=4)

    # activation：对某些点额外标色（区分数值型与颜色字符串/元组）
    if activation_dict:
        for idx in activation_dict.keys():
            p = activation_dict[idx]
            if effective_dim == 3:
                ax.scatter(p[0], p[1], p[2], s=30, marker='o', color='red')
            else:
                ax.scatter(p[0], p[1], s=30, marker='o', color='red')

    plt.colorbar(sc, label=name)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def scatter_f_on_domain(domain, f, name="f", tdim=3, title="Function on Domain", activation_dict=None):
    val = eval_function(f, domain.geometry.x)
    scatter_val_on_domain(domain, val, name=name, tdim=tdim, title=title, activation_dict=activation_dict)

def plot_activation_times_on_mesh(mesh_file, act_times, gdim=3, target_cell=2, title="Activation Times on Mesh", activation_dict=None):
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain
    f = Function(functionspace(domain, ("Lagrange", 1)))
    f.x.array[:] = act_times
    scatter_f_on_domain(domain, f, name="Activation Time", tdim=domain.topology.dim, title=title, activation_dict=activation_dict)

def plot_loss_and_cm(loss_per_iter, cm_cmp_per_iter):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(loss_per_iter)
    plt.title('cost functional')
    plt.xlabel('iteration')
    plt.subplot(1, 2, 2)
    plt.plot(cm_cmp_per_iter)
    plt.title('error in center of mass')
    plt.xlabel('iteration')
    plt.show()

def plot_scatter_on_mesh(mesh_file, gdim=3, target_cell=2, title="Mesh Plot", scatter_pts=None, point_size=12, cmap="viridis"):
    # --- 创建 pyvista 网格 ---
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain

    
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, gdim))
    # --- 初始化 plotter ---
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_title(title)
    plotter.view_yz()
    plotter.add_axes()

    # --- 绘制散点 ---
    if scatter_pts is not None:# 固定颜色（红色）
        for pts in scatter_pts:
            plotter.add_points(
                pts,
                color="red",
                point_size=point_size,
                render_points_as_spheres=True
            )

    # --- 展示 ---
    plotter.show(auto_close=False)

def plot_vals_on_mesh(
    mesh_file,
    val_2d,                 # shape = (T, N)
    gdim=3,
    target_cell=None,
    name="f",
    title="Function on Mesh",
    f_val_flag=False,
    n_rows=3,
    n_cols=4,
    step_per_timeframe=4
):
    """
    自动选择合适时间间隔，绘制多个时刻。
    val_2d: shape = (T, N)
    """

    # ----------- 读 mesh ------------
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)

    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain

    # ----------- 数据检查 ------------
    T, N = val_2d.shape
    max_plots = n_rows * n_cols

    if len(domain.geometry.x) != N:
        raise ValueError(f"val_2d 列数 N={N} 与网格点数 {len(domain.geometry.x)} 不一致！")

    # ----------- 自动选择 T 间隔 ------------
    if T <= max_plots:
        selected_indices = np.arange(T)
    else:
        interval = T // max_plots
        selected_indices = np.arange(0, interval * max_plots, interval)

    # print("Selected time indices:", selected_indices)  # Debug 可开启

    # ----------- 创建 plotter ------------
    plotter = pyvista.Plotter(shape=(n_rows, n_cols), border=False)

    # ---- 生成基准网格 ----
    grid_src = pyvista.UnstructuredGrid(*vtk_mesh(domain, domain.topology.dim))

    # ----------- 绘制每一帧 ------------
    for i, t in enumerate(selected_indices):
        r = i // n_cols
        c = i % n_cols
        plotter.subplot(r, c)

        grid = grid_src.copy()

        if f_val_flag:
            f = Function(functionspace(domain, ("Lagrange", 1)))
            f.x.array[:] = val_2d[t]
            val = eval_function(f, domain.geometry.x)
        else:
            val = val_2d[t]

        grid.point_data[name] = val
        grid.set_active_scalars(name)

        plotter.add_mesh(grid, show_edges=True)
        plotter.add_title(f"{title} (t={t/step_per_timeframe}s)")
        plotter.view_yz()
        plotter.add_axes()

    plotter.show(auto_close=False)