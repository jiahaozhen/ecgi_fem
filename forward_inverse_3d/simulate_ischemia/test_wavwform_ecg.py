from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import plot_bsp_on_standard12lead, plot_v_random

if __name__ == '__main__':
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    # mesh_file = r'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_40_lc_ratio_5.msh'
    T = 500
    step_per_timeframe = 8
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         T=T, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         ischemia_flag=False)
    d_data = compute_d_from_tmp(mesh_file, v_data)
    
    plot_bsp_on_standard12lead(d_data,
                               step_per_timeframe=step_per_timeframe,
                               filter_flag=False)
    plot_v_random(v_data, step_per_timeframe=step_per_timeframe)
