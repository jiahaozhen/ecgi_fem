# Using tmp data from reaction_diffusion to compute bsp
# forward : FEM, BEM
# save the result in .mat format
import sys

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from main_forward_tmp import compute_d_from_tmp
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.helper_function import eval_function, transfer_bsp_to_standard12lead

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ischemia_flag = False
step_per_timeframe = 10
v_data_fem, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, gdim=3, T=500, 
                                                             step_per_timeframe=step_per_timeframe, 
                                                             ischemia_flag=ischemia_flag)
d_data_fem = compute_d_from_tmp(mesh_file, v_data=v_data_fem)

geom_data_ecgsim = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])

V = functionspace(subdomain_ventricle, ("Lagrange", 1))
v = Function(V)
v_data_bem = []
total_num = v_data_fem.shape[0]
for i in range(total_num):
    v.x.array[:] = v_data_fem[i]
    v_surface = eval_function(v, points=v_pts_ecgsim).reshape(-1)
    v_data_bem.append(v_surface.copy())
v_data_bem = np.array(v_data_bem)

forward_matrix = np.array(geom_data_ecgsim['ventricles2thorax'])
d_data_bem = v_data_bem @ forward_matrix

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_fem = transfer_bsp_to_standard12lead(d_data_fem, leadIndex)
standard12Lead_bem = transfer_bsp_to_standard12lead(d_data_bem, leadIndex)

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_1 = np.arange(0, standard12Lead_fem.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
time_2 = np.arange(0, standard12Lead_bem.shape[0] / step_per_timeframe, 1 / step_per_timeframe)
for i, ax in enumerate(axs.flat):
    ax.plot(time_1, standard12Lead_fem[:, i])
    ax.plot(time_2, standard12Lead_bem[:, i], linestyle='--')
    ax.legend(['fem', 'bem'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# if ischemia_flag:
#     sio.savemat(r'forward_inverse_3d/data/ischemia_reaction_fem.mat', {'surface_potential': d_data_fem})
#     sio.savemat(r'forward_inverse_3d/data/ischemia_reaction_bem.mat', {'surface_potential': d_data_bem, 'tmp': v_data_bem})
# else:
#     sio.savemat(r'forward_inverse_3d/data/normal_reaction_fem.mat', {'surface_potential': d_data_fem})
#     sio.savemat(r'forward_inverse_3d/data/normal_reaction_bem.mat', {'surface_potential': d_data_bem, 'tmp': v_data_bem})