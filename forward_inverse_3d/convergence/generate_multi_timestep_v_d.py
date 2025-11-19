import sys
import numpy as np
import os
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_ischemia import compute_d_from_tmp

if __name__ == "__main__":
    # 参数设置
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    timestep_list = [2, 4, 8, 16]  # 新增时间步长列表

    for timestep in timestep_list:
        step_per_timeframe = timestep  # 修改 step_per_timeframe 为当前 timestep
        output_file = f'forward_inverse_3d/data/convergence_results/results_timestep_{timestep}.npz'

        if os.path.exists(output_file):
            print(f"Results for timestep={timestep} already exist. Skipping computation.")
            continue

        print(f"==== Running timestep = {timestep} ====")

        # 1️⃣ 计算膜电位分布
        v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=500, step_per_timeframe=step_per_timeframe)

        # 2️⃣ 计算体表电位
        d_data = compute_d_from_tmp(mesh_file, v_data)

        # 保存结果
        np.savez(output_file, v_data=v_data, d_data=d_data)
        print(f"Results saved to {output_file}")