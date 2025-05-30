import numpy as np

from main_ischemia_if_activation_known import ischemia_inversion

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
v_data_file = '3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy'
d_data_file = '3d/data/u_data_reaction_diffusion_ischemia_data_argument_30dB.npy'
phi_1_file = "3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
phi_2_file = "3d/data/phi_2_data_reaction_diffusion_ischemia_20dB.npy"
v_data = np.load(v_data_file)
d_data = np.load(d_data_file)
phi_1_exact = np.load(phi_1_file)
phi_2_exact = np.load(phi_2_file)

time_sequence_1 = np.arange(0, 50, 2) 
phi_1_rest = ischemia_inversion(mesh_file=mesh_file, d_data=d_data, v_data=v_data, 
                                phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                                time_sequence=time_sequence_1, 
                                alpha1=1e-2, total_iter=200,
                                plot_flag=True, print_message=True, transmural_flag=True)

time_sequence_2 = np.arange(0, 1200, 10)
phi_1_activation = ischemia_inversion(mesh_file=mesh_file, d_data=d_data, v_data=v_data,
                                      phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                                      time_sequence=time_sequence_2, phi_initial=phi_1_rest.x.array,
                                      alpha1=1e-2, total_iter=50,
                                      plot_flag=True, print_message=True, transmural_flag=False)