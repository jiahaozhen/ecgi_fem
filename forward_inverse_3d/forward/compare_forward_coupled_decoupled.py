# 基于有限元正过程比较心脏躯干解耦耦合模型结果
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from forward_inverse_3d.forward.forward_coupled import compute_d_from_tmp as compute_d_from_tmp_coupled
from forward_inverse_3d.forward.forward_decoupled import compute_d_from_tmp as compute_d_from_tmp_decoupled
from utils.visualize_tools import compare_bsp_on_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
T = 500
step_per_timeframe = 2

v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=False, T=T, step_per_timeframe=step_per_timeframe)

d_data_coupled = compute_d_from_tmp_coupled(mesh_file, v_data, multi_flag=True)
d_data_decoupled = compute_d_from_tmp_decoupled(mesh_file, v_data, multi_flag=True)

compare_bsp_on_standard12lead(d_data_coupled, d_data_decoupled, 
                              labels=['Coupled Model', 'Decoupled Model'], 
                              step_per_timeframe=step_per_timeframe, filter_flag=False)