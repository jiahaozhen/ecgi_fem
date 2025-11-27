import numpy as np
from forward_inverse_3d.forward.forward_coupled import forward_tmp
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.helper_function import add_noise_based_on_snr

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
ischemia_flag = False
ischemia_epi_endo=[-1, 0, 1]
center_ischemia=np.array([32.1, 71.7, 15])
radius_ischemia=30
v_data, phi_1, phi_2 = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                             ischemia_flag=ischemia_flag,
                                                             ischemia_epi_endo=ischemia_epi_endo,
                                                             center_ischemia=center_ischemia,
                                                             radius_ischemia=radius_ischemia)

# sample data
u_data, _ = forward_tmp(mesh_file, v_data)

# save data
file_name_prefix = 'forward_inverse_3d/data/inverse/'
file_name_suffix = '_data_reaction_diffusion'

u_file_name = (file_name_prefix + 'u' + file_name_suffix
               + ('_ischemia' if ischemia_flag else '_normal')
               + '.npy')
v_file_name = (file_name_prefix + 'v' + file_name_suffix 
               + ('_ischemia' if ischemia_flag else '_normal')
               + '.npy')

u_data_10dB = add_noise_based_on_snr(u_data, snr=10)
u_data_20dB = add_noise_based_on_snr(u_data, snr=20)
u_data_30dB = add_noise_based_on_snr(u_data, snr=30)

np.save(u_file_name, u_data)
np.save(v_file_name, v_data)
np.save(u_file_name.replace('.npy', '_10dB.npy'), u_data_10dB)
np.save(u_file_name.replace('.npy', '_20dB.npy'), u_data_20dB)
np.save(u_file_name.replace('.npy', '_30dB.npy'), u_data_30dB)