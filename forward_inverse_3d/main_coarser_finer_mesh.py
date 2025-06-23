import sys
import os
from main_create_mesh_ecgsim_multi_conduct import create_mesh
from main_forward_tmp import forward_tmp
from main_ischemia_if_activation_known import ischemia_inversion
import numpy as np
sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.helper_function import extract_data_from_mesh1_to_mesh2, extract_data_from_submesh1_to_submesh2, add_noise_based_on_snr
# generate coarser and finer meshes
file_name_prefix = '3d/data/finer_coarser_mesh/'
coarser_mesh_file = file_name_prefix + 'mesh_multi_conduct_ecgsim_coarser.msh'
finer_mesh_file = file_name_prefix + 'mesh_multi_conduct_ecgsim_finer.msh'
if not os.path.exists(coarser_mesh_file):
    create_mesh(coarser_mesh_file, lc=60, multi_flag=True)
if not os.path.exists(finer_mesh_file):
    create_mesh(finer_mesh_file, lc=40, multi_flag=True)

# forward problem on finer mesh
u_coarser_file = file_name_prefix + 'u_coarser.npy'
v_coarser_file = file_name_prefix + 'v_coarser.npy'
phi_1_coarser_file = file_name_prefix + 'phi_1_coarser.npy'
phi_2_coarser_file = file_name_prefix + 'phi_2_coarser.npy'
if os.path.exists(u_coarser_file) and os.path.exists(v_coarser_file) and \
   os.path.exists(phi_1_coarser_file) and os.path.exists(phi_2_coarser_file):
    u_data_coarser = np.load(u_coarser_file)
    v_data_coarser = np.load(v_coarser_file)
    phi_1_coarser = np.load(phi_1_coarser_file)
    phi_2_coarser = np.load(phi_2_coarser_file)
else:
    # transmembrane potential data
    ischemia_flag = True
    data_argument = True
    v_data_finer, phi_1_finer, phi_2_finer = compute_v_based_on_reaction_diffusion(
                                                                finer_mesh_file, tau=10,
                                                                ischemia_flag=ischemia_flag, 
                                                                data_argument=data_argument)
    # body surface potential data
    u_data_finer = forward_tmp(finer_mesh_file, v_data_finer)

    # extract data from finer mesh to coarser mesh
    u_data_coarser = extract_data_from_mesh1_to_mesh2(finer_mesh_file, coarser_mesh_file, u_data_finer)
    phi_1_coarser = extract_data_from_submesh1_to_submesh2(finer_mesh_file, coarser_mesh_file, phi_1_finer)
    phi_2_coarser = extract_data_from_submesh1_to_submesh2(finer_mesh_file, coarser_mesh_file, phi_2_finer)
    v_data_coarser = extract_data_from_submesh1_to_submesh2(finer_mesh_file, coarser_mesh_file, v_data_finer)

    # save data
    np.save(file_name_prefix + 'u_coarser.npy', u_data_coarser)
    np.save(file_name_prefix + 'v_coarser.npy', v_data_coarser)
    np.save(file_name_prefix + 'phi_1_coarser.npy', phi_1_coarser)
    np.save(file_name_prefix + 'phi_2_coarser.npy', phi_2_coarser)

u_data_coarser = add_noise_based_on_snr(u_data_coarser, snr=30)

# inverse problem on coarser mesh
time_sequence_1 = np.arange(0, 50, 2)
phi_1_rest = ischemia_inversion(mesh_file=coarser_mesh_file, d_data=u_data_coarser, v_data=v_data_coarser, 
                                phi_1_exact=phi_1_coarser, phi_2_exact=phi_2_coarser,
                                time_sequence=time_sequence_1, 
                                alpha1=1e-2, total_iter=200,
                                plot_flag=True, print_message=True, transmural_flag=True)
time_sequence_2 = np.arange(0, 1200, 10)
phi_1_activation = ischemia_inversion(mesh_file=coarser_mesh_file, d_data=u_data_coarser, v_data=v_data_coarser,
                                       phi_1_exact=phi_1_coarser, phi_2_exact=phi_2_coarser,
                                       time_sequence=time_sequence_2, phi_initial=phi_1_rest.x.array,
                                       alpha1=1e-2, total_iter=50,
                                       plot_flag=True, print_message=True, transmural_flag=False)
