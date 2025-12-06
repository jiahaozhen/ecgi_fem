import time
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp as compute_d_coupled_matrix
from forward_inverse_3d.forward.forward_coupled_ischemia import compute_d_from_tmp as compute_d_coupled_ischemia
from forward_inverse_3d.forward.forward_coupled import compute_d_from_tmp as compute_d_coupled
from forward_inverse_3d.forward.forward_ecgsim import compute_d_from_tmp as compute_d_ecgsim
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.simulate_tools import get_activation_dict
from utils.visualize_tools import compare_bsp_on_standard12lead

def test_forward_processes():
    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    case_name = case_name_list[0]
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
    T = 500
    step_per_timeframe = 8

    activation_dict = get_activation_dict(case_name, mode='ENDO', threshold=40)

    # Generate v_data using a common method
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         T=T, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         ischemia_flag=False,
                                                         tau_in_val=0.4,
                                                         activation_dict_origin=activation_dict)

    # Test forward_coupled_matrix_from
    start_time = time.time()
    d_coupled_matrix = compute_d_coupled_matrix(case_name, v_data, allow_cache=True)
    time_coupled_matrix = time.time() - start_time


    # Test forward_coupled
    start_time = time.time()
    d_coupled = compute_d_coupled(case_name, v_data)
    time_coupled = time.time() - start_time

    # Print results
    print("Comparison of Forward Processes:")
    print(f"Coupled Matrix: Time = {time_coupled_matrix:.4f} seconds")
    print(f"Coupled: Time = {time_coupled:.4f} seconds")

    print("Visualizing and comparing 12-lead results...")
    
    compare_bsp_on_standard12lead(d_coupled_matrix, d_coupled, 
                                  labels=["Coupled Matrix", "Coupled"],
                                  step_per_timeframe=step_per_timeframe,
                                  filter_flag=False, filter_window_size=step_per_timeframe*10)
    
if __name__ == "__main__":
    test_forward_processes()