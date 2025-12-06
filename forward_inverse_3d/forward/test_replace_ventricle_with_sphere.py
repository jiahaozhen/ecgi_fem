import numpy as np
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import plot_bsp_on_standard12lead, plot_activation_times_on_mesh, plot_v_random
from utils.transmembrane_potential_tools import get_activation_time_from_v

if __name__ == "__main__":
    # 读取网格
    case_name = 'normal_male[sphere]'
    mesh_file = f"forward_inverse_3d/data/mesh/mesh_{case_name}.msh"
    step_per_timeframe = 16

    # 确认激活序列

    activation_dict = {
        8 : np.array([57, 51.2, 15]),
        14.4 : np.array([30.2, 45.2, -30]),
        14.5 : np.array([12.8, 54.2, -15]),
        18.7 : np.array([59.4, 29.8, 15]),
        23.5 : np.array([88.3, 41.2, -37.3]),
        34.9 : np.array([69.1, 27.1, -30]),
        45.6 : np.array([48.4, 40.2, -37.5])
    }

    # 确认跨膜电压
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=500, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         activation_dict_origin=activation_dict,
                                                         tau_in_val=0.4)

    # 确认心电图
    d_data = compute_d_from_tmp(case_name, v_data, multi_flag=False, allow_cache=False)
    act_time = get_activation_time_from_v(v_data) / step_per_timeframe
    
    import multiprocessing

    p1 = multiprocessing.Process(target=plot_bsp_on_standard12lead, 
                                 args=(d_data,), 
                                 kwargs={'step_per_timeframe':step_per_timeframe, 
                                         'filter_flag':False, 
                                         'filter_window_size':step_per_timeframe*10})
    p2 = multiprocessing.Process(target=plot_v_random, 
                                 args=(v_data,), 
                                 kwargs={'step_per_timeframe':step_per_timeframe})
    p3 = multiprocessing.Process(target=plot_activation_times_on_mesh, 
                                 args=(mesh_file, act_time), 
                                 kwargs={'activation_dict':activation_dict})
    
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
