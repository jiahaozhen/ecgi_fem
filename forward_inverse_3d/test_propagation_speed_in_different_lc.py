import sys

from main_create_mesh_ecgsim_ventricle import create_mesh

sys.path.append('.')
from reaction_diffusion.main_reaction_diffusion_on_ventricle import compute_v_based_on_reaction_diffusion
from utils.helper_function import get_activation_time_from_v

activation_last = []
for lc in range(2, 15):
    target_file = f'3d/data/mesh_ecgsim_ventricle_{lc}.msh'
    # if the file exists, skip the mesh creation
    try:
        with open(target_file):
            pass
    except IOError:
        create_mesh(target_file, lc)
    v_data_0_1 = compute_v_based_on_reaction_diffusion(target_file)
    time = max(get_activation_time_from_v(v_data_0_1))/5
    print(f'lc: {lc}, time: {time} ms')
    activation_last.append(time)
print(activation_last)