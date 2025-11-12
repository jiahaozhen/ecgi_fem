import sys
import numpy as np
import os

sys.path.append('.')

from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp

if __name__ == "__main__":
    # 参数设置
    mesh_file_template = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'
    step_per_timeframe = 16
    lc_list = [20, 40, 80, 160]
    lc_ratio_list = [5]
    T = 500

    for lc in lc_list:
        for lc_ratio in lc_ratio_list:
            mesh_file = mesh_file_template.format(lc, lc_ratio)

            if not os.path.exists(mesh_file):
                print(f"Mesh file not found: {mesh_file}. Skipping.")
                continue

            output_file = f'forward_inverse_3d/data/convergence_results/results_lc_{lc}_ratio_{lc_ratio}.npz'

            if os.path.exists(output_file):
                print(f"Results for lc={lc}, lc_ratio={lc_ratio} already exist. Skipping computation.")
                continue

            print(f"==== Running lc = {lc}, lc_ratio = {lc_ratio} ====")

            # 1️⃣ 计算膜电位分布
            import time
            start_time = time.time()
            v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe)
            end_time = time.time()
            print(f"Membrane potential computation time: {end_time - start_time:.2f} seconds")

            # 2️⃣ 计算体表电位
            d_data = compute_d_from_tmp(mesh_file, v_data)
            end_time = time.time()
            print(f"Body surface potential computation time: {end_time - start_time:.2f} seconds")

            # 保存结果
            np.savez(output_file, v_data=v_data, d_data=d_data)
            print(f"Results saved to {output_file}")
