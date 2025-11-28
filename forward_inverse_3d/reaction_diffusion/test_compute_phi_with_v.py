from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.simulate_tools import get_activation_dict
from utils.transmembrane_potential_tools import compute_phi_with_v
from utils.visualize_tools import plot_val_on_mesh

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe = 8

    activation_dict = get_activation_dict(mesh_file, threshold=40)

    v_data, marker, f_space = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         ischemia_flag=True, 
                                                         T=500, 
                                                         step_per_timeframe=step_per_timeframe,
                                                         activation_dict_origin=activation_dict)
    
    import time
    start_time = time.time()
    phi_1, phi_2 = compute_phi_with_v(v_data, marker, f_space)
    end_time = time.time()
    print(f"Time taken to compute phi: {end_time - start_time} seconds")

    plot_val_on_mesh(mesh_file, phi_2[80], target_cell=2, f_val_flag=True, title="Phi 1 at first timeframe")