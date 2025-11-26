# Using tmp data from reaction_diffusion to compute bsp
# forward : FEM, BEM

from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp as compute_d_from_tmp_fem
from forward_inverse_3d.forward.forward_ecgsim import compute_d_from_tmp as compute_d_from_tmp_bem
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.visualize_tools import compare_bsp_on_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'

ischemia_flag = False
step_per_timeframe = 8
v_data_fem, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, gdim=3, T=500, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         ischemia_flag=ischemia_flag)

d_data_fem = compute_d_from_tmp_fem(mesh_file, v_data=v_data_fem)
d_data_bem = compute_d_from_tmp_bem(mesh_file, v_data=v_data_fem)

compare_bsp_on_standard12lead(d_data_fem, d_data_bem,
                              labels = ['FEM', 'BEM'],
                              step_per_timeframe=step_per_timeframe,
                              filter_flag=False)