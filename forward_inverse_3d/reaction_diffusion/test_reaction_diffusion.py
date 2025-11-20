import time
import numpy as np
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import plot_v_random
from utils.simulate_tools import get_activation_dict

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe = 4
    
    # activation_dict = {
    #     8 : np.array([57, 51.2, 15]),
    #     14.4 : np.array([30.2, 45.2, -30]),
    #     14.5 : np.array([12.8, 54.2, -15]),
    #     18.7 : np.array([59.4, 29.8, 15]),
    #     23.5 : np.array([88.3, 41.2, -37.3]),
    #     34.9 : np.array([69.1, 27.1, -30]),
    #     45.6 : np.array([48.4, 40.2, -37.5])
    # }

    activation_dict = get_activation_dict(mesh_file, threshold=40)

    start_time = time.time()
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         ischemia_flag=True, 
                                                         T=500, 
                                                         step_per_timeframe=step_per_timeframe,
                                                         activation_dict_origin=activation_dict)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    plot_v_random(v_data, step_per_timeframe=step_per_timeframe)