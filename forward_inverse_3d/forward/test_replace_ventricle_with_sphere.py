import numpy as np
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import plot_bsp_on_standard12lead, plot_v_random

if __name__ == "__main__":
    # 读取网格
    mesh_file = r"forward_inverse_3d/data/mesh/mesh_ecgsim_sphere.msh"
    step_per_timeframe = 16

    # 确认激活序列

    activation_dict = {
        5: np.array([46, 39, -10])
    }

    # 确认跨膜电压
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=400, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         activation_dict_origin=activation_dict,
                                                         tau_in_val=0.8)

    # 确认心电图
    d_data = compute_d_from_tmp(mesh_file, v_data, multi_flag=False, allow_cache=False)
    plot_bsp_on_standard12lead(d_data, step_per_timeframe=step_per_timeframe,
                               filter_flag=False, filter_window_size=step_per_timeframe*10)
    # 可视化跨膜电压
    plot_v_random(v_data, step_per_timeframe=step_per_timeframe)
