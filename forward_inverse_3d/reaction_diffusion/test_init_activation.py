from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import plot_scatter_on_mesh
from utils.simulate_tools import get_activation_dict
from utils.transmembrane_potential_tools import get_activation_time_from_v

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe = 8

    activation_dict = get_activation_dict(mesh_file, threshold=10)

    # start_time = time.time()
    # v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
    #                                                      ischemia_flag=True, 
    #                                                      T=500, 
    #                                                      step_per_timeframe=step_per_timeframe,
    #                                                      activation_dict_origin=activation_dict)
    # end_time = time.time()
    # print(f"Simulation time: {end_time - start_time} seconds")

    plot_scatter_on_mesh(mesh_file, scatter_pts=list(activation_dict.values()))