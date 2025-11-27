import os
import numpy as np
import logging
import time
from utils.machine_learning_tools import project_bsp_on_surface, load_bsp_pts

def save_partial_bsp_data(surface_d_results, seg_ids, save_dir, partial_idx):
    """
    Save partial BSP data and corresponding segment IDs to a compressed file.

    Parameters:
        bsp_results (list): List of BSP data arrays.
        seg_ids (list): List of segment IDs corresponding to BSP data.
        save_dir (str): Directory to save the data.
        partial_idx (int): Index for the partial file.
    """
    surface_d_array = np.array(surface_d_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"ischemia_d_surface_part_{partial_idx:03d}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(partial_file, X=surface_d_array, y=seg_ids_array)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset/d_surface_dataset'

    # Load v_data and seg_ids from ischemia dataset
    d_standard_data_dir = 'machine_learning/data/dataset/d_standard_dataset'
    d_standard_data_files = [os.path.join(d_standard_data_dir, f) for f in os.listdir(d_standard_data_dir) if f.endswith('.npz')]
    d_standard_data_files.sort()  # Ensure consistent order
    d_standard_data = []
    seg_ids = []

    original_pts = load_bsp_pts()

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_standard_data_files):
        with np.load(file) as data:
            d_standard_data = data['X']
            seg_ids = data['y']


        surface_d_results = []
        surface_d_seg_ids = []

        for i, (d, seg_id) in enumerate(zip(d_standard_data, seg_ids)):
            try:
                start_time = time.time()
                surface_d = project_bsp_on_surface(d)
                surface_d_results.append(surface_d)
                surface_d_seg_ids.append(seg_id)
                elapsed_time = time.time() - start_time
                logging.info(f"✅ 已处理数据索引 {i}，耗时 {elapsed_time:.2f} 秒")
                print(f"Processed {i+1}/{len(d_standard_data)} in file {file_idx+1}/{len(d_standard_data_files)}", end='\r')
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if surface_d_results:
            save_partial_bsp_data(surface_d_results, 
                                  surface_d_seg_ids, 
                                  save_dir, file_idx)
            logging.info(f"✅ 已处理文件 {file}")