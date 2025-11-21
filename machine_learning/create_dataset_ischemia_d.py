import os
import numpy as np
import logging
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp

def save_partial_bsp_data(bsp_results, seg_ids, save_dir, partial_idx):
    """
    Save partial BSP data and corresponding segment IDs to a compressed file.

    Parameters:
        bsp_results (list): List of BSP data arrays.
        seg_ids (list): List of segment IDs corresponding to BSP data.
        save_dir (str): Directory to save the data.
        partial_idx (int): Index for the partial file.
    """
    bsp_array = np.array(bsp_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"ischemia_d_part_{partial_idx:03d}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(partial_file, X=bsp_array, y=seg_ids_array)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset/d_dataset'

    # Load v_data and seg_ids from ischemia dataset
    v_data_dir = 'machine_learning/data/dataset/v_dataset'
    v_data_files = [os.path.join(v_data_dir, f) for f in os.listdir(v_data_dir) if f.endswith('.npz')]
    v_data_files.sort()  # Ensure consistent order
    v_data = []
    seg_ids = []

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(v_data_files):
        with np.load(file) as data:
            v_data = data['X']
            seg_ids = data['y']

        bsp_results = []
        bsp_seg_ids = []

        for i, (v, seg_id) in enumerate(zip(v_data, seg_ids)):
            try:
                bsp = compute_d_from_tmp(mesh_file, v, allow_cache=True)  # Simplified call
                bsp_results.append(bsp)
                bsp_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if bsp_results:
            save_partial_bsp_data(bsp_results, bsp_seg_ids, save_dir, file_idx)
            logging.info(f"✅ 已处理文件 {file}")