import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_one_timeframe import resting_ischemia_inversion

if __name__ == '__main__':
    mesh_file = r"forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh"
    d = np.load(r'forward_inverse_3d/data/inverse/u_data_reaction_diffusion_ischemia.npy')[0]
    v = np.load(r'forward_inverse_3d/data/inverse/v_data_reaction_diffusion_ischemia.npy')[0]
    resting_ischemia_inversion(mesh_file, d_data=d, v_data=v, plot_flag=True, print_message=True, transmural_flag=True)