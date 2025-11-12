import numpy as np
from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.simulate_ischemia.forward_coupled import compute_d_from_tmp
from utils.visualize_tools import compare_standard_12_lead
from utils.helper_function import transfer_bsp_to_standard12lead

def test_D_tau_Mi_Me_effect():
    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe = 4
    T = 500

    # Define different parameter combinations to test
    test_cases = [
        {'affect_D': False, 'affect_tau_in': True, 'affect_tau_close': True, 'affect_Mi': True, 'affect_M': True, 'case': 'No D effect'},
        {'affect_D': True, 'affect_tau_in': False, 'affect_tau_close': True, 'affect_Mi': True, 'affect_M': True, 'case': 'No tau_in effect'},
        {'affect_D': True, 'affect_tau_in': True, 'affect_tau_close': False, 'affect_Mi': True, 'affect_M': True, 'case': 'No tau_close effect'},
        {'affect_D': True, 'affect_tau_in': True, 'affect_tau_close': True, 'affect_Mi': False, 'affect_M': True, 'case': 'No Mi effect'},
        {'affect_D': True, 'affect_tau_in': True, 'affect_tau_close': True, 'affect_Mi': True, 'affect_M': False, 'case': 'No M effect'},
        {'affect_D': True, 'affect_tau_in': True, 'affect_tau_close': True, 'affect_Mi': True, 'affect_M': True, 'case': 'All effects'},
        {'affect_D': False, 'affect_tau_in': False, 'affect_tau_close': False, 'affect_Mi': False, 'affect_M': False, 'case': 'No effects'},
    ]

    results = []
    labels = []

    for i, params in enumerate(test_cases):
        print(f"Running test case {i+1} with parameters: {params}")
        v_data, _, _ = compute_v_based_on_reaction_diffusion(
            mesh_file,
            ischemia_flag=True,
            T=T,
            step_per_timeframe=step_per_timeframe,
            affect_D=params['affect_D'],
            affect_tau_in=params['affect_tau_in'],
            affect_tau_close=params['affect_tau_close']
        )
        d_data = compute_d_from_tmp(
            mesh_file,
            v_data,
            ischemia_flag=True,
            affect_Mi=params['affect_Mi'],
            affect_M=params['affect_M']
        )
        stand_12_lead = transfer_bsp_to_standard12lead(d_data, 
                                                       lead_index=np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1)
        results.append(stand_12_lead)
        labels.append(params['case'])

    # Compare results using the visualization tool
    compare_standard_12_lead(*results, 
                             labels=labels, 
                             step_per_timeframe=step_per_timeframe, 
                             filter_flag=False)

if __name__ == "__main__":
    test_D_tau_Mi_Me_effect()