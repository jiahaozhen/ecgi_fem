from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import plot_bsp_on_standard12lead, plot_v_random
from utils.simulate_tools import get_activation_dict, transform_v_into_ecgsim_form

if __name__ == '__main__':
    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    case_name = case_name_list[0]
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
    T = 500
    step_per_timeframe = 8
    
    activation_dict = get_activation_dict(case_name, mode='ENDO', threshold=40)

    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file,
                                                         T=T,
                                                         step_per_timeframe=step_per_timeframe,
                                                         activation_dict_origin=activation_dict)
    
    v_data = transform_v_into_ecgsim_form(v_data, step_per_timeframe)
    plot_v_random(v_data, step_per_timeframe)
    d_data = compute_d_from_tmp(case_name, v_data, allow_cache=True)
    
    plot_bsp_on_standard12lead(d_data,
                               step_per_timeframe=step_per_timeframe,
                               filter_flag=False)