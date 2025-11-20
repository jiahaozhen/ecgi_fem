'''
测试 tau_close 在空间中是否变化对 12 导联心电图的影响
结论 tau_close 的变化对 12 导联心电图影响较大 结果会偏离正常范围
'''
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import compare_bsp_on_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 500
step_per_timeframe = 4

v_data_no_vary, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe, tau_close_vary=False)
v_data_vary, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, T=T, step_per_timeframe=step_per_timeframe, tau_close_vary=True)

d_data_no_vary = compute_d_from_tmp(mesh_file, v_data_no_vary)
d_data_vary = compute_d_from_tmp(mesh_file, v_data_vary)

compare_bsp_on_standard12lead(d_data_no_vary, d_data_vary,
                              labels=['No tau_close vary', 'tau_close vary'],
                              step_per_timeframe=step_per_timeframe,
                              filter_flag=False)