import sys

import numpy as np

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.helper_function import get_activation_time_from_v

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe_list = [1,2,4,5,10,20]
    activation_duration_dict = {}
    for step_per_timeframe in step_per_timeframe_list:
        u_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=150, step_per_timeframe=step_per_timeframe)
        activation_time = get_activation_time_from_v(u_data)
        max_activation = np.max(activation_time[np.isfinite(activation_time)])
        min_activation = np.min(activation_time[np.isfinite(activation_time)])
        activation_duration_dict[step_per_timeframe] = (max_activation - min_activation) / step_per_timeframe
        print(f"Activation time range for step_per_timeframe={step_per_timeframe}: {(max_activation - min_activation) / step_per_timeframe} ms")

    # plot convergence of activation duration
    import matplotlib.pyplot as plt
    plt.figure()
    x = step_per_timeframe_list
    y = [activation_duration_dict[step] for step in step_per_timeframe_list]
    plt.plot(x, y, marker='o')
    plt.xlabel('Step per Timeframe')
    plt.ylabel('Activation Duration (ms)')
    plt.title('Test-Time Convergence of Activation Duration')
    plt.grid()
    plt.show()