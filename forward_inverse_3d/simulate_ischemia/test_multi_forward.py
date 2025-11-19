import time
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp as compute_d_coupled_matrix
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp as compute_d_coupled
from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import compare_bsp_on_standard12lead

def test_forward_processes():
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    T = 500
    step_per_timeframe = 8

    # Generate v_data using a common method
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         T=T, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         ischemia_flag=False)

    # Test forward_coupled_matrix_from
    start_time = time.time()
    d_coupled_matrix = compute_d_coupled_matrix(mesh_file, v_data)
    time_coupled_matrix = time.time() - start_time


    # Test forward_coupled
    start_time = time.time()
    d_coupled = compute_d_coupled(mesh_file, v_data)
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