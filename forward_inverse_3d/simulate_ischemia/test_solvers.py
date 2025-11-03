import time
import numpy as np
from forward_inverse_3d.simulate_ischemia.forward_coupled import forward_tmp
from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from petsc4py import PETSc
import matplotlib.pyplot as plt
from utils.helper_function import transfer_bsp_to_standard12lead

def test_different_solvers(mesh_file, v_data, solver_configs=None):
    """
    测试不同求解器的性能和结果。

    Parameters:
    - mesh_file: str, 网格文件路径
    - v_data: ndarray, 输入膜电位数据
    - solver_configs: list of dict, 每个字典包含求解器类型和预条件器配置
      e.g., [{'type': 'cg', 'pc': 'ilu'}, {'type': 'gmres', 'pc': 'jacobi'}]

    Returns:
    - results: dict, 包含每种求解器的运行时间和结果差异
    """
    if solver_configs is None:
        solver_configs = [
            {'type': 'cg', 'pc': 'hypre'},
            {'type': 'cg', 'pc': 'ilu'},
            {'type': 'gmres', 'pc': 'ilu'},
            {'type': 'cg', 'pc': 'gamg'}
        ]

    results = {}

    for config in solver_configs:
        solver_type = config['type']
        pc_type = config['pc']
        print(f"\nTesting solver: {solver_type} with PC: {pc_type}")

        # 设置求解器
        solver = PETSc.KSP().create()
        solver.setType(solver_type)
        solver.getPC().setType(pc_type)

        # 记录时间
        start_time = time.time()

        # 运行 forward_tmp
        try:
            d_data, _ = forward_tmp(mesh_file, v_data, solver=solver)
            elapsed_time = time.time() - start_time

            # 保存结果
            results[f"{solver_type}_{pc_type}"] = {
                'time': elapsed_time,
                'd_data': d_data
            }

            print(f"Solver {solver_type} with PC {pc_type} completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            print(f"Solver {solver_type} with PC {pc_type} failed: {e}")
            results[f"{solver_type}_{pc_type}"] = {
                'time': None,
                'error': str(e)
            }

    # 比较结果差异
    base_solver = list(results.keys())[0]
    base_data = results[base_solver]['d_data'] if 'd_data' in results[base_solver] else None

    for solver_key, data in results.items():
        if solver_key != base_solver and 'd_data' in data:
            diff = np.linalg.norm(base_data - data['d_data'])
            results[solver_key]['difference'] = diff
            print(f"Difference between {base_solver} and {solver_key}: {diff:.6f}")

    return results

def plot_12_lead_comparison(results, lead_indices):
    """
    按照标准 12 导联格式绘制不同求解器的比较结果。

    Parameters:
    - results: dict, 包含每种求解器的运行结果
    - lead_indices: list, 12 导联的电极索引（0-based）
    """
    plt.figure(figsize=(15, 10))

    for i in range(12):
        plt.subplot(4, 3, i + 1)
        for solver, data in results.items():
            if 'd_data' in data:
                lead_data = transfer_bsp_to_standard12lead(data['d_data'], lead_indices)[:, i]
                plt.plot(lead_data, label=f"Solver: {solver}")

        plt.title(f"Lead {i + 1}")
        plt.xlabel("Time Frame")
        plt.ylabel("Potential (mV)")
        plt.grid(True)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    T = 400
    lead_indices = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1  # 12 导联索引

    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T, step_per_timeframe=2)

    solver_configs = [
            {'type': 'cg', 'pc': 'hypre'},
            {'type': 'cg', 'pc': 'ilu'},
            {'type': 'gmres', 'pc': 'ilu'},
            {'type': 'cg', 'pc': 'gamg'},
            {'type': 'preonly', 'pc': 'ilu'}
        ]
    solver_results = test_different_solvers(mesh_file, v_data, solver_configs=solver_configs)

    # 绘制 12 导联比较图
    plot_12_lead_comparison(solver_results, lead_indices)