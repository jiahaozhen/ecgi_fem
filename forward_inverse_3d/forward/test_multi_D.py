from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import plot_bsp_on_standard12lead
from utils.simulate_tools import get_activation_dict

if __name__ == '__main__':
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    T = 500
    step_per_timeframe = 8
    
    activation_dict = get_activation_dict(mesh_file, mode='ENDO', threshold=40)

    import numpy as np

    # 设置 D 的测试范围
    D_val_list = np.arange(0.1, 1.1, 0.1)

    for D in D_val_list:
        print(f"Testing D = {D}")
        v_data, _, _ = compute_v_based_on_reaction_diffusion(
            mesh_file,
            T=T,
            step_per_timeframe=step_per_timeframe,
            activation_dict_origin=activation_dict,
            D_val=D
        )
        d_data = compute_d_from_tmp(mesh_file, v_data, allow_cache=True)
        plot_bsp_on_standard12lead(d_data, 
                                   step_per_timeframe=step_per_timeframe, 
                                   filter_flag=False)

    