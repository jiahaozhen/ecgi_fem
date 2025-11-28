import os
import numpy as np
from dolfinx.io import gmshio
from dolfinx.fem import functionspace
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from utils.signal_processing_tools import transfer_bsp_to_standard12lead
from utils.visualize_tools import compare_standard_12_lead, plot_convergence
from utils.error_metrics_tools import compute_convergence_metrics

# =========================================================
# 基础配置
# =========================================================
mesh_file_template = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'
lc_list = [10, 20, 40, 80]
lc_ratio_list = [1, 2, 4, 8]

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

domain_target, cell_markers_target, _ = gmshio.read_from_msh('forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_160_lc_ratio_5.msh', MPI.COMM_WORLD)
subdomain_ventricle_target, _, _, _ = create_submesh(domain_target, 3, cell_markers_target.find(2))
target_pts = subdomain_ventricle_target.geometry.x

def extract_v_data(mesh_file, v_data):

    # 读取网格文件
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD)
    tdim = domain.topology.dim

    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    # 创建函数空间
    V = functionspace(subdomain_ventricle, ("CG", 1))
    coords = V.tabulate_dof_coordinates()

    distances = np.linalg.norm(coords - target_pts[:, np.newaxis], axis=2)
    closest_vertex = np.argmin(distances, axis=1)

    return v_data[:, closest_vertex]

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    # 更新后的文件路径
    filename_template = 'forward_inverse_3d/data/convergence_results/results_lc_{}_ratio_{}.npz'

    v_data_dict = {}
    d_data_dict = {}

    # 找到存在的文件中 lc 最小且 lc_ratio 最大的基准值
    base_lc, base_lc_ratio = None, None
    for lc in sorted(lc_list):
        for lc_ratio in sorted(lc_ratio_list, reverse=False):
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
    for lc, lc_ratio in zip(lc_list, lc_ratio_list):
        filename = filename_template.format(lc, lc_ratio)
        try:
            data = np.load(filename)
            v_data_dict[(lc, lc_ratio)] = extract_v_data(mesh_file_template.format(lc, lc_ratio), data['v_data'])  # 确保数据格式正确
            d_data_dict[(lc, lc_ratio)] = transfer_bsp_to_standard12lead(data['d_data'], leadIndex)  # 确保数据格式正确
            print(f"Loaded data for lc={lc}, lc_ratio={lc_ratio} from {filename}")
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping.")

    # 如果没有数据，退出程序
    if not v_data_dict or not d_data_dict:
        print("No data files found. Exiting.")
        exit()

    # 计算指标并绘图
    v_metrics_summary = compute_convergence_metrics(v_data_dict, base_lc=base_lc, base_lc_ratio=base_lc_ratio)
    d_metrics_summary = compute_convergence_metrics(d_data_dict, base_lc=base_lc, base_lc_ratio=base_lc_ratio)

    import multiprocessing

    p1 = multiprocessing.Process(target=plot_convergence, args=(v_metrics_summary, base_lc, base_lc_ratio))
    p2 = multiprocessing.Process(target=plot_convergence, args=(d_metrics_summary, base_lc, base_lc_ratio))
    p3 = multiprocessing.Process(target=compare_standard_12_lead, args=(*d_data_dict.values(),),
                                        kwargs={'labels': [f"lc={lc}, ratio={lc_ratio}" for (lc, lc_ratio) in sorted(d_data_dict.keys())],
                                                'step_per_timeframe': 4,
                                                'filter_flag': False})
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
