import sys
import numpy as np
import os

sys.path.append('.')

from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp

if __name__ == "__main__":
    # 参数设置
    mesh_file_template = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'
    step_per_timeframe = 2
    lc_list = [20, 30, 40, 50, 60, 70, 80]
    lc_ratio_list = [2, 3, 4, 5]

    # 9 个导联电极索引（转为 0-based）
    leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

    for lc in lc_list:
        for lc_ratio in lc_ratio_list:
            mesh_file = mesh_file_template.format(lc, lc_ratio)

            if not os.path.exists(mesh_file):
                print(f"Mesh file not found: {mesh_file}. Skipping.")
                continue

            output_file = f'forward_inverse_3d/data/results_lc_{lc}_ratio_{lc_ratio}.npz'

            if os.path.exists(output_file):
                print(f"Results for lc={lc}, lc_ratio={lc_ratio} already exist. Skipping computation.")
                continue

            print(f"==== Running lc = {lc}, lc_ratio = {lc_ratio} ====")

            # 1️⃣ 计算膜电位分布
            v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=500, step_per_timeframe=step_per_timeframe)

            # 2️⃣ 计算体表电位
            d_data = compute_d_from_tmp(mesh_file, v_data)

            # 保存结果
            np.savez(output_file, v_data=v_data, d_data=d_data)
            print(f"Results saved to {output_file}")
