import sys

import numpy as np
from main_forward_tmp import forward_tmp
from main_create_mesh_ecgsim_multi_conduct import create_mesh
from main_ischemia_time_related import ischemia_inversion

sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion
from utils.helper_function import compute_cc

gdim = 3
mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
v_peak_i_0_1 = 0.7
v_rest_i_0_1 = 0.3
print('create mesh...')
create_mesh(mesh_file, lc = 40)
print('compute v based on reaction diffusion...')
v_min, v_max = -90, 10
v_data = compute_v_based_on_reaction_diffusion(
    mesh_file=mesh_file, T = 40, 
    u_peak_ischemia_val=v_peak_i_0_1, u_rest_ischemia_val=v_rest_i_0_1,
    submesh_flag=True, ischemia_flag=True,
    data_argument=False, v_max=v_max, v_min=v_min
)

# sample data
# v_data = v_data[::5]
print('forward...')
u_data = forward_tmp(mesh_file, v_data, gdim = gdim)
u_data += np.random.normal(0, 0.1, u_data.shape)

print('inverse...')
for alpha1 in [1e-1, 1e0, 1e1]:
    for alpha2 in [1e0]:
        for alpha3 in [1e1]:
            for tau in [1e0, 5e0, 1e1]:
                phi_1, phi_2, v_result = ischemia_inversion(mesh_file=mesh_file, d_data=u_data, v_exact=v_data, gdim=gdim,
                                    a1=v_min, a2=(v_max - v_min) * v_rest_i_0_1 + v_min, 
                                    a3=v_max, a4=(v_max - v_min) * v_peak_i_0_1 + v_min,
                                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, tau=tau
                                    )
                print('alpha1:', alpha1, 'alpha2:', alpha2, 'alpha3:', alpha3)
                cc_v = []
                for i in range(v_data.shape[0]):
                    # cc of v_exact and v_result
                    cc_v.append(np.corrcoef(v_data[i], v_result[i])[0, 1])
                cc = np.array(cc_v)
                print('cc of v_data and v_result:', np.mean(cc))