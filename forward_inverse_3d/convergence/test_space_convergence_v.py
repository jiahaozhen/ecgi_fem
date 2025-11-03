import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils.helper_function import transfer_bsp_to_standard12lead

# =========================================================
# 基础配置
# =========================================================
mesh_file_template = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'
lc_list = [20, 30, 40, 50, 60, 70, 80]
lc_ratio_list = [2, 3, 4, 5]

def extract_v_data(mesh_file, v_data):
    from dolfinx.io import gmshio
    from dolfinx.fem import functionspace
    from dolfinx.mesh import create_submesh
    from mpi4py import MPI

    # 读取网格文件
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD)
    tdim = domain.topology.dim

    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    # 创建函数空间
    V = functionspace(subdomain_ventricle, ("CG", 1))

    target_pts = np.array([[59.4, 29.8, 15], [88.3, 41.2, -37.3], [69.1, 27.1, -30]])
    coords = V.tabulate_dof_coordinates()

    distances = np.linalg.norm(coords - target_pts[:, np.newaxis], axis=2)
    closest_vertex = np.argmin(distances, axis=1)

    return v_data[:, closest_vertex]


# =========================================================
# 指标计算函数
# =========================================================
def compute_metrics_for_v(v_base, v_other):
    """
    计算单个 lc 和 lc_ratio 相对于基准的多指标相似性。
    v_base, v_other: shape = (n_leads, n_time)
    返回:
        {
          "corr": 平均皮尔逊相关,
          "rmse": 均方根误差,
          "rel_L2": 相对L2误差,
          "peak_shift": 平均峰值时间偏移（采样点数）
        }
    """
    n_leads = v_base.shape[1]
    r_list, rmse_list, relL2_list, peak_shift_list = [], [], [], []

    for i in range(n_leads):
        x, y = v_base[:,i], v_other[:,i]

        # 1. 皮尔逊相关系数
        if np.std(x) == 0 or np.std(y) == 0:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)

        # 2. RMSE
        rmse = np.sqrt(np.mean((x - y) ** 2))

        # 3. 相对L2误差
        rel_L2 = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)

        # 4. 峰值时间偏移
        peak_shift = np.argmax(y) - np.argmax(x)

        r_list.append(r)
        rmse_list.append(rmse)
        relL2_list.append(rel_L2)
        peak_shift_list.append(peak_shift)

    metrics = {
        "corr": np.nanmean(r_list),
        "rmse": np.mean(rmse_list),
        "rel_L2": np.mean(relL2_list),
        "peak_shift": np.mean(np.abs(peak_shift_list))
    }
    return metrics

# =========================================================
# 主指标计算
# =========================================================
def compute_v_convergence_metrics(v_data_dict, base_lc=20, base_lc_ratio=1):
    """
    计算不同 lc 和 lc_ratio 下 V 信号相对基准的多指标相似度
    """
    base_v = v_data_dict[(base_lc, base_lc_ratio)]
    summary = {}

    print("\n=== Mesh Convergence Analysis for V ===")
    for (lc, lc_ratio), v in sorted(v_data_dict.items()):
        if (lc, lc_ratio) == (base_lc, base_lc_ratio):
            continue

        metrics = compute_metrics_for_v(base_v, v)
        summary[(lc, lc_ratio)] = metrics
        print(
            f"lc={lc:>3d}, lc_ratio={lc_ratio:>2d} vs base lc={base_lc}, lc_ratio={base_lc_ratio}: "
            f"corr={metrics['corr']:.3f}, "
            f"relL2={metrics['rel_L2']:.3e}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"Δt_peak={metrics['peak_shift']:.2f}"
        )

    return summary

# =========================================================
# 绘图函数
# =========================================================
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

# =========================================================
# 绘制 v_data_dict 数据的函数
# =========================================================
def plot_v_data_dict(v_data_dict):
    """
    将 v_data_dict 中不同 lc 和 lc_ratio 的数据绘制在一张图上，
    但不同的 v 数据绘制在不同的子图中。
    """
    num_v = next(iter(v_data_dict.values())).shape[1]  # 获取 v 的数量

    fig, axs = plt.subplots(num_v, 1, figsize=(10, 3 * num_v), sharex=True)  # 调整每张子图的大小
    if num_v == 1:
        axs = [axs]  # 保证 axs 是一个列表

    for v_idx in range(num_v):
        for (lc, lc_ratio), v_data in sorted(v_data_dict.items()):
            axs[v_idx].plot(v_data[:,v_idx], label=f"lc={lc}, lc_ratio={lc_ratio}")
        axs[v_idx].set_title(f"V Index {v_idx}")
        axs[v_idx].set_xlabel("Time")
        axs[v_idx].set_ylabel("V Values")
        axs[v_idx].legend()
        axs[v_idx].grid(True)

    plt.tight_layout()
    plt.show()

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    # 更新后的文件路径
    filename_template = 'forward_inverse_3d/data/results_lc_{}_ratio_{}.npz'

    v_data_dict = {}

    # 找到存在的文件中 lc 最小且 lc_ratio 最大的基准值
    base_lc, base_lc_ratio = None, None
    for lc in sorted(lc_list):
        for lc_ratio in sorted(lc_ratio_list, reverse=True):
            filename = filename_template.format(lc, lc_ratio)
            if os.path.exists(filename):
                base_lc, base_lc_ratio = lc, lc_ratio
                break
        if base_lc is not None:
            break

    if base_lc is None or base_lc_ratio is None:
        print("No valid mesh files found. Exiting.")
        exit()
    
    print(f"Default base values: lc={base_lc}, lc_ratio={base_lc_ratio}")

    # 加载每个 lc 和 lc_ratio 的数据
    for lc in lc_list:
        for lc_ratio in lc_ratio_list:
            filename = filename_template.format(lc, lc_ratio)
            try:
                data = np.load(filename)
                v_data_dict[(lc, lc_ratio)] = extract_v_data(mesh_file_template.format(lc, lc_ratio), data['v_data'])  # 确保数据格式正确
                print(f"Loaded data for lc={lc}, lc_ratio={lc_ratio} from {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping.")

    # 如果没有数据，退出程序
    if not v_data_dict:
        print("No data files found. Exiting.")
        exit()

    # 计算指标并绘图
    metrics_summary = compute_v_convergence_metrics(v_data_dict, base_lc=base_lc, base_lc_ratio=base_lc_ratio)
    plot_convergence(metrics_summary, base_lc=base_lc, base_lc_ratio=base_lc_ratio)

    # 绘制 v_data_dict 数据
    plot_v_data_dict(v_data_dict)