# 将d转化为标准化数据（电压差）
import os
import numpy as np
import logging
from utils.signal_processing_tools import transfer_bsp_to_standard64lead

def save_partial_bsp_data(standard_d_results, seg_ids, save_dir, partial_idx):
    """
    Save partial BSP data and corresponding segment IDs to a compressed file.

    Parameters:
        bsp_results (list): List of BSP data arrays.
        seg_ids (list): List of segment IDs corresponding to BSP data.
        save_dir (str): Directory to save the data.
        partial_idx (int): Index for the partial file.
    """
    standard_d_array = np.array(standard_d_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"ischemia_d_standard_part_{partial_idx:03d}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(partial_file, X=standard_d_array, y=seg_ids_array)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset/d64_standard_dataset'

    # Load v_data and seg_ids from ischemia dataset
    d_data_dir = 'machine_learning/data/dataset/d_dataset'
    d_data_files = [os.path.join(d_data_dir, f) for f in os.listdir(d_data_dir) if f.endswith('.npz')]
    d_data_files.sort()  # Ensure consistent order
    d_data = []
    seg_ids = []

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_data_files):
        with np.load(file) as data:
            d_data = data['X']
            seg_ids = data['y']

        standard_d_results = []
        standard_d_seg_ids = []

        for i, (d, seg_id) in enumerate(zip(d_data, seg_ids)):
            try:
                standard_d = transfer_bsp_to_standard64lead(d)
                standard_d_results.append(standard_d)
                standard_d_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if standard_d_results:
            save_partial_bsp_data(standard_d_results, 
                                  standard_d_seg_ids, 
                                  save_dir, file_idx)
            logging.info(f"✅ 已处理文件 {file}")