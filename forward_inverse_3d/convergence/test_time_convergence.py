import numpy as np
import matplotlib.pyplot as plt
from utils.signal_processing_tools import transfer_bsp_to_standard12lead
from utils.error_metrics_tools import compute_metrics_for_timestep

# =========================================================
# 基础配置
# =========================================================
result_file_template = 'forward_inverse_3d/data/convergence_results/results_timestep_{}.npz'
timestep_list = [2, 4, 8, 16]
T = 500
leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

def compute_timestep_convergence_metrics(data_dict, base_timestep):
    """
    计算不同时间步长下数据相对基准的多指标相似度。

    参数：
        data_dict (dict): 包含不同时间步长的数据。
        base_timestep (int): 基准时间步长。

    返回：
        summary (dict): 每个时间步长的收敛性指标。
    """
    base_data = data_dict[base_timestep]
    summary = {}

    for timestep, data in sorted(data_dict.items()):
        if timestep == base_timestep:
            continue

        metrics = compute_metrics_for_timestep(base_data, data)
        summary[timestep] = metrics

    return summary

# =========================================================
# 绘图函数
# =========================================================
def plot_timestep_convergence(summary, base_timestep=1):
    """
    绘制时间步长收敛性图（仅相关系数）
    """
    timesteps = [str(timestep) for timestep in sorted(summary.keys())]
    corr_vals = [summary[key]['corr'] for key in sorted(summary.keys())]

    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, corr_vals, 'o-', label='Correlation (r)')
    plt.title(f"Mean Correlation vs Timestep (base timestep={base_timestep})")
    plt.xlabel("Timestep")
    plt.ylabel("Correlation (r)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_convergence_comparison(v_summary, d_summary, base_timestep):
    """
    绘制 v_data 和 d_data 的收敛性对比图
    """
    timesteps = [str(timestep) for timestep in sorted(v_summary.keys())]
    v_corr_vals = [v_summary[key]['corr'] for key in sorted(v_summary.keys())]
    d_corr_vals = [d_summary[key]['corr'] for key in sorted(d_summary.keys())]

    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, v_corr_vals, 'o-', label='V Data Correlation (r)')
    plt.plot(timesteps, d_corr_vals, 's--', label='D Data Correlation (r)')
    plt.title(f"Convergence Comparison (Base Timestep={base_timestep})")
    plt.xlabel("Timestep")
    plt.ylabel("Correlation (r)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_12lead_comparison(d_data_dict):
    """
    绘制十二导联叠加图，不同时间步长的数据叠加。

    参数：
        d_data_dict: dict, 包含不同时间步长的 d_data 数据。
        base_timestep: int, 基准时间步长。
    """
    leads = [
        "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
        "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
    ]

    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        for timestep, d_data in sorted(d_data_dict.items()):
            time = np.linspace(0, T, d_data.shape[0])  # 假设时间归一化
            ax.plot(time, d_data[:, i], label=f"timestep={timestep}")

        ax.set_title(leads[i])
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Potential (mV)")
        ax.grid(True)
        ax.legend(fontsize="small")

    fig.suptitle(f"12-lead ECG Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    d_data_dict = {}
    d_data_12lead_dict = {}
    v_data_dict = {}

    # 找到存在的文件中时间步长最小的基准值
    base_timestep = 16
    for timestep in timestep_list:
        filename = result_file_template.format(timestep)
        try:
            data = np.load(filename)

            # 加载 d_data
            d_data = data['d_data']
            d_data_dict[timestep] = d_data  # 保留全部电压数据用于指标计算

            # 转换为12导联格式用于作图
            d_data_12lead = transfer_bsp_to_standard12lead(d_data, leadIndex)
            d_data_12lead_dict[timestep] = d_data_12lead

            # 加载 v_data
            v_data = data['v_data']
            v_data_dict[timestep] = v_data

            print(f"Loaded data for timestep={timestep} from {filename}")
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping.")

    # 如果没有数据，退出程序
    if not d_data_dict or not v_data_dict:
        print("No data files found. Exiting.")
        exit()

    # 计算 v_data 和 d_data 的指标
    v_metrics_summary = compute_timestep_convergence_metrics(v_data_dict, base_timestep=base_timestep)
    d_metrics_summary = compute_timestep_convergence_metrics(d_data_dict, base_timestep=base_timestep)

    # 输出 v_data 和 d_data 的指标
    print("\n=== V Data Convergence Metrics ===")
    for timestep, metrics in v_metrics_summary.items():
        print(
            f"[V Data] timestep={timestep:>3d} vs base timestep={base_timestep}: "
            f"corr={metrics['corr']:.3f}"
        )

    print("\n=== D Data Convergence Metrics ===")
    for timestep, metrics in d_metrics_summary.items():
        print(
            f"[D Data] timestep={timestep:>3d} vs base timestep={base_timestep}: "
            f"corr={metrics['corr']:.3f}"
        )

    import multiprocessing

    p1 = multiprocessing.Process(target=plot_convergence_comparison, args=(v_metrics_summary, d_metrics_summary, base_timestep))
    p2 = multiprocessing.Process(target=plot_12lead_comparison, args=(d_data_dict, ))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()