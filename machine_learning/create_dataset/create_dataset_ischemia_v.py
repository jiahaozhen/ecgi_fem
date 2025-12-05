"""
创建心肌缺血数据集
输入： 多时刻体表电压数据
输出： 缺血位置分区数据

变化维度：
1. 缺血位置： N(心室节点数)
2. 缺血半径： 2个半径(10, 20)
3. 缺血范围： 3个范围(外膜, 内膜, 外中内)
4. 严重程度： 2个程度(-80mV/0mV, -70mV/-10mV)
"""

import os
import logging
import h5py

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from tqdm import tqdm
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh
from utils.simulate_tools import get_activation_dict


# Function to generate ischemia data
def generate_ischemia_data(
    mesh_file,
    save_dir,
    gdim=3,
    T=500,
    step_per_timeframe=1,
    save_interval=200,
    partial_idx=0,
):
    os.makedirs(save_dir, exist_ok=True)
    logging.info("开始生成心肌缺血数据集")

    activation_dict = get_activation_dict(mesh_file, mode='ENDO', threshold=40)

    segment_ids, _, _ = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    valid_mask = segment_ids != -1
    center_ischemia_list = subdomain_ventricle.geometry.x[valid_mask]
    center_segment_ids = segment_ids[valid_mask]

    # 参数定义
    radius_ischemia_list = [0, 15, 30]
    ischemia_epi_endo_list = [[1, 0], [0, 1], [-1, 0, 1]]
    u_peak_ischemia_val_list = [0.9, 0.8]
    u_rest_ischemia_val_list = [0.1, 0.2]
    all_v_results = []
    all_seg_ids = []

    total_loops = (
        len(center_ischemia_list)
        * len(radius_ischemia_list)
        * len(ischemia_epi_endo_list)
        * len(u_peak_ischemia_val_list)
    )

    with tqdm(total=total_loops, desc="生成心肌缺血数据集", dynamic_ncols=True) as pbar:
        for center_ischemia, seg_id in zip(center_ischemia_list, center_segment_ids):
            for radius_ischemia in radius_ischemia_list:
                for ischemia_epi_endo in ischemia_epi_endo_list:
                    for u_peak_ischemia_val, u_rest_ischemia_val in zip(
                        u_peak_ischemia_val_list, u_rest_ischemia_val_list
                    ):
                        try:
                            v, _, _ = compute_v_based_on_reaction_diffusion(
                                mesh_file,
                                gdim=gdim,
                                ischemia_flag=True,
                                ischemia_epi_endo=ischemia_epi_endo,
                                center_ischemia=center_ischemia,
                                radius_ischemia=radius_ischemia,
                                T=T,
                                step_per_timeframe=step_per_timeframe,
                                u_peak_ischemia_val=u_peak_ischemia_val,
                                u_rest_ischemia_val=u_rest_ischemia_val,
                                activation_dict_origin=activation_dict,
                            )
                            all_v_results.append(v)
                            all_seg_ids.append(seg_id)
                            pbar.update(1)

                            if len(all_v_results) >= save_interval:
                                save_partial_data(
                                    all_v_results, all_seg_ids, save_dir, partial_idx
                                )
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
    partial_file = os.path.join(save_dir, f"ischemia_v_part_{partial_idx:03d}.h5")
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_ischemia_data(
        mesh_file='forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh',
        save_dir='machine_learning/data/dataset_dl/v_dataset',
        # save_dir='machine_learning/data/dataset_ml/v_dataset',
    )
