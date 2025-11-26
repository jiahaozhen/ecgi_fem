import numpy as np
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import plot_bsp_on_standard12lead, plot_activation_times_on_mesh, plot_v_random, plot_val_on_mesh
from utils.helper_function import get_activation_time_from_v

if __name__ == "__main__":
    # 读取网格
    mesh_file = r"forward_inverse_3d/data/mesh/mesh_ecgsim_multi_sphere.msh"
    step_per_timeframe = 16

    # 确认激活序列

    # activation_dict = {
    #     8.1 : np.array([46.8, 9.3, -40.9]),
    #     8.2 : np.array([46.8, 70.1, -40.9])
    # }

    activation_dict = {
        8: np.array([46.8, 39.7, -40.9])
    }

    # 确认跨膜电压
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=500, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         activation_dict_origin=activation_dict)

    # 确认心电图
    d_data = compute_d_from_tmp(mesh_file, v_data, multi_flag=True, allow_cache=True)
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
    p4 = multiprocessing.Process(target=plot_val_on_mesh, 
                                 args=(mesh_file, v_data[20*step_per_timeframe]), 
                                 kwargs={'target_cell':2, 'f_val_flag':True})
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()