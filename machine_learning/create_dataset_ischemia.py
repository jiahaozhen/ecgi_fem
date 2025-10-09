"""
创建心肌缺血数据集
输入： 多时刻体表电压数据
输出： 缺血位置分区数据

变化维度：
1. 缺血位置： N(心室节点数)
2. 缺血半径： 3个半径(10, 20, 30)
3. 缺血范围： 5个范围(外膜, 内膜, 外中, 内中, 外中内)
4. 严重程度： 3个程度(-80mV/0mV, -70mV/-10mV, -60mV/-20mV)
"""

import sys
import os

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from tqdm import tqdm

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh
from forward_inverse_3d.main_forward_tmp import compute_d_from_tmp

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
save_dir = r'machine_learning/dataset/ischemia_partial'
os.makedirs(save_dir, exist_ok=True)

gdim = 3
T = 120
step_per_timeframe = 2
save_interval = 200   # 每多少个样本保存一次
partial_idx = 0      # 从第几个文件编号开始（断点续算时改这里）


segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
center_ischemia_list = subdomain_ventricle.geometry.x

radius_ischemia_list = [10, 20, 30]
ischemia_epi_endo_list = [[1, 0], [-1, 0], [-1, 0, 1]]
u_peak_ischemia_val_list = [0.9, 0.8, 0.7]
u_rest_ischemia_val_list = [0.1, 0.2, 0.3]

# v
all_v_results = []
all_seg_ids = []

valid_centers = [c for c, s in zip(center_ischemia_list, segment_ids) if s != -1]
total_loops = len(valid_centers) * len(radius_ischemia_list) * len(ischemia_epi_endo_list) * len(u_peak_ischemia_val_list)

with tqdm(total=total_loops, desc="生成心肌缺血数据集", dynamic_ncols=True) as pbar:

    for center_ischemia, seg_id in zip(center_ischemia_list, segment_ids):
        if seg_id == -1:
            continue
        for radius_ischemia in radius_ischemia_list:
            for ischemia_epi_endo in ischemia_epi_endo_list:
                for u_peak_ischemia_val, u_rest_ischemia_val in zip(u_peak_ischemia_val_list, u_rest_ischemia_val_list):
                    v, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, gdim=gdim,
                                                            ischemia_flag=True, ischemia_epi_endo=ischemia_epi_endo,
                                                            center_ischemia=center_ischemia, radius_ischemia=radius_ischemia,
                                                            T=T, step_per_timeframe=step_per_timeframe,
                                                            u_peak_ischemia_val=u_peak_ischemia_val, u_rest_ischemia_val=u_rest_ischemia_val,
                                                            v_min=-90, v_max=10)
                    all_v_results.append(v)
                    all_seg_ids.append(seg_id)
                    pbar.update(1)

                    if len(all_v_results) >= save_interval:
                        n_sample = len(all_v_results)
                        all_v_results = np.vstack(all_v_results)
                        all_d_results = compute_d_from_tmp(mesh_file, all_v_results)
                        all_d_results = np.split(all_d_results, n_sample, axis=0)

                        X = np.array(all_d_results)
                        y = np.array(all_seg_ids)

                        partial_file = os.path.join(save_dir, f"ischemia_part_{partial_idx:03d}.npz")
                        np.savez_compressed(partial_file, X=X, y=y)
                        print(f"✅ 已保存 {partial_file}")

                        # 清空缓存
                        all_v_results = []
                        all_seg_ids = []
                        partial_idx += 1

if len(all_v_results) > 0:
    n_sample = len(all_v_results)
    all_v_results = np.vstack(all_v_results)
    all_d_results = compute_d_from_tmp(mesh_file, all_v_results)
    all_d_results = np.split(all_d_results, n_sample, axis=0)

    X = np.array(all_d_results)
    y = np.array(all_seg_ids)

    partial_file = os.path.join(save_dir, f"ischemia_part_{partial_idx:03d}.npz")
    np.savez_compressed(partial_file, X=X, y=y)
    print(f"✅ 已保存最后文件 {partial_file}")