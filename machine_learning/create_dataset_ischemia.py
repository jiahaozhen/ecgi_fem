"""
创建心肌缺血数据集
输入： 多时刻体表电压数据
输出： 缺血位置分区数据

变化维度：
1. 缺血位置： N(心室节点数)
2. 缺血半径： 3个半径(10, 20, 30)
3. 缺血范围： 5个范围(外膜, 内膜, 外中, 内中, 外中内)
4. 严重程度： 2个程度(-80mV/0mV, -70mV/-10mV)
"""

import os
import logging

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from tqdm import tqdm
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh

# Function to generate ischemia data
def generate_ischemia_data(mesh_file, save_dir, gdim=3, T=500, step_per_timeframe=2, save_interval=200, partial_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    logging.info("开始生成心肌缺血数据集")

    segment_ids, _, _ = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    center_ischemia_list = []
    for seg_id in np.unique(segment_ids):
        if seg_id == -1:
            continue
        segment_points = subdomain_ventricle.geometry.x[segment_ids == seg_id]
        if len(segment_points) > 5:
            selected_points = segment_points[np.random.choice(len(segment_points), 5, replace=False)]
        else:
            selected_points = segment_points
        center_ischemia_list.extend(selected_points)

    # 参数定义
    radius_ischemia_list = [10, 20]
    ischemia_epi_endo_list = [[1, 0], [-1, 0], [-1, 0, 1]]
    u_peak_ischemia_val_list = [0.9, 0.8]
    u_rest_ischemia_val_list = [0.1, 0.2]

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
                        try:
                            v, _, _ = compute_v_based_on_reaction_diffusion(
                                mesh_file, gdim=gdim, ischemia_flag=True, ischemia_epi_endo=ischemia_epi_endo,
                                center_ischemia=center_ischemia, radius_ischemia=radius_ischemia,
                                T=T, step_per_timeframe=step_per_timeframe,
                                u_peak_ischemia_val=u_peak_ischemia_val, u_rest_ischemia_val=u_rest_ischemia_val,
                                v_min=-90, v_max=10
                            )
                            all_v_results.append(v)
                            all_seg_ids.append(seg_id)
                            pbar.update(1)

                            if len(all_v_results) >= save_interval:
                                save_partial_data(all_v_results, all_seg_ids, save_dir, partial_idx)
                                partial_idx += 1
                                all_v_results.clear()
                                all_seg_ids.clear()
                        except Exception as e:
                            logging.error(f"数据生成失败: {e}")

    if all_v_results:
        save_partial_data(all_v_results, all_seg_ids, save_dir, partial_idx)
        logging.info("✅ 已保存最后文件")

# Function to save partial data
def save_partial_data(v_results, seg_ids, save_dir, partial_idx):
    X = np.array(v_results)
    y = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"ischemia_part_{partial_idx:03d}.npz")
    np.savez_compressed(partial_file, X=X, y=y)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_ischemia_data(
        mesh_file='machine_learning/data/mesh/mesh_multi_conduct_ecgsim.msh',
        save_dir='machine_learning/data/dataset/ischemia_partial'
    )