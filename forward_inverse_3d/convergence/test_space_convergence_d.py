import os
import numpy as np
from utils.helper_function import transfer_bsp_to_standard12lead
from utils.visualize_tools import compare_standard_12_lead, plot_convergence
from utils.error_metrics_tools import compute_convergence_metrics

# =========================================================
# 基础配置
# =========================================================
mesh_file_template = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'

lc_list = [20, 40, 80, 160]
lc_ratio_list = [5]

# 9 个导联电极索引（转为 0-based）
leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    # 更新后的文件路径
    filename_template = 'forward_inverse_3d/data/convergence_results/results_lc_{}_ratio_{}.npz'

    d_data_dict = {}

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
                d_data_dict[(lc, lc_ratio)] = transfer_bsp_to_standard12lead(data['d_data'], leadIndex)  # 确保数据格式正确
                print(f"Loaded data for lc={lc}, lc_ratio={lc_ratio} from {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping.")

    # 如果没有数据，退出程序
    if not d_data_dict:
        print("No data files found. Exiting.")
        exit()

    # 计算指标并绘图
    metrics_summary = compute_convergence_metrics(d_data_dict, base_lc=base_lc, base_lc_ratio=base_lc_ratio)

    import multiprocessing

    p1 = multiprocessing.Process(target=plot_convergence, args=(metrics_summary, base_lc, base_lc_ratio))
    p2 = multiprocessing.Process(target=compare_standard_12_lead, args=(*d_data_dict.values(),), kwargs={
        'labels': [f"lc={lc}, ratio={lc_ratio}" for (lc, lc_ratio) in sorted(d_data_dict.keys())],
        'step_per_timeframe': 16,
        'filter_flag': False
    })
    p1.start()
    p2.start()

    p1.join()
    p2.join()
