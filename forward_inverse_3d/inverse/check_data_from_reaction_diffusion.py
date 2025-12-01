'''
检查reaction-diffusion方程生成的数据
'''
import numpy as np
import multiprocessing
from utils.visualize_tools import plot_vals_on_mesh, plot_v_random

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
v_exact_all_time = np.load('forward_inverse_3d/data/inverse/v_data_ischemia.npy')[0:400]
phi_1_exact = np.load('forward_inverse_3d/data/inverse/phi_1_data_ischemia.npy')[0:400]
phi_2_exact = np.load('forward_inverse_3d/data/inverse/phi_2_data_ischemia.npy')[0:400]

p1 = multiprocessing.Process(target=plot_vals_on_mesh, kwargs={"mesh_file": mesh_file,
                                                               "val_2d": v_exact_all_time,
                                                               "target_cell": 2,
                                                               "f_val_flag": True,
                                                               "title": "v"})
p2 = multiprocessing.Process(target=plot_vals_on_mesh, kwargs={"mesh_file": mesh_file,
                                                               "val_2d": phi_1_exact,
                                                               "target_cell": 2,
                                                               "f_val_flag": True,
                                                               "title": "phi_1"})
p3 = multiprocessing.Process(target=plot_vals_on_mesh, kwargs={"mesh_file": mesh_file,
                                                               "val_2d": phi_2_exact,
                                                               "target_cell": 2,
                                                               "f_val_flag": True,
                                                               "title": "phi_2"})
p4 = multiprocessing.Process(target=plot_v_random, kwargs={"v_data": v_exact_all_time})

p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()