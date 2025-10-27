import sys

import numpy as np

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.helper_function import get_activation_time_from_v

if __name__ == "__main__":
    mesh_file = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}.msh'
    step_per_timeframe_list = 5
    lc_list = [20, 40, 60, 80]
    activation_duration_dict = {}
    for lc in lc_list:
        u_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file.format(lc), T=150, step_per_timeframe=step_per_timeframe_list)
        activation_time = get_activation_time_from_v(u_data)
        max_activation = np.max(activation_time[np.isfinite(activation_time)])
        min_activation = np.min(activation_time[np.isfinite(activation_time)])
        activation_duration_dict[lc] = (max_activation - min_activation) / step_per_timeframe_list
        print(f"Activation time range for lc={lc}: {(max_activation - min_activation) / step_per_timeframe_list} ms")
    
    # plot convergence of activation duration
    import matplotlib.pyplot as plt
    plt.figure()
    x = lc_list
    y = [activation_duration_dict[lc] for lc in lc_list]
    plt.plot(x, y, marker='o')
    plt.xlabel('Characteristic Length (lc)')
    plt.ylabel('Activation Duration (ms)')
    plt.title('Test-Space Convergence of Activation Duration')
    plt.grid()
    plt.show()