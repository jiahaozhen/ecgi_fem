import os
import numpy as np
import logging
from tqdm import tqdm  # Import tqdm for progress bar
import gc  # Import garbage collection module
import psutil  # Import psutil to monitor memory usage
from forward_inverse_3d.simulate_ischemia.forward_ecgsim import compute_d_from_tmp

def process_forward_tmp_output(v_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.info("开始处理 forward_tmp 输出数据")

    # Process data file-by-file and row-by-row to minimize memory usage
    with tqdm(total=len(v_data_files), desc="Processing files", dynamic_ncols=True) as file_pbar:
        for file_idx, file in enumerate(v_data_files):
            with np.load(file) as data:
                v_data = data['X']
                seg_ids = data['y']

            for i, (v, seg_id) in enumerate(zip(v_data, seg_ids)):
                try:
                    bsp = compute_d_from_tmp(mesh_file, v)  # Simplified call
                    save_partial_bsp_data([bsp], [seg_id], save_dir, file_idx * 1000 + i)
                except Exception as e:
                    logging.error(f"处理数据失败: {e}")

                # Monitor memory usage
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 90:  # If memory usage exceeds 90%
                    logging.warning("High memory usage detected. Consider reducing chunk size further.")

            file_pbar.update(1)

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
    partial_file = os.path.join(save_dir, f"bsp_part_{partial_idx:03d}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(partial_file, bsp=bsp_array, seg_ids=seg_ids_array)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset/bsp_partial'

    # Load v_data and seg_ids from ischemia dataset
    v_data_dir = 'machine_learning/data/dataset/ischemia_partial'
    v_data_files = [os.path.join(v_data_dir, f) for f in os.listdir(v_data_dir) if f.endswith('.npz')]
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
                bsp = compute_d_from_tmp(mesh_file, v)  # Simplified call
                bsp_results.append(bsp)
                bsp_seg_ids.append(seg_id)

                if len(bsp_results) >= 200:  # Fixed save interval
                    save_partial_bsp_data(bsp_results, bsp_seg_ids, save_dir, file_idx * 1000 + i // 200)
                    bsp_results.clear()
                    bsp_seg_ids.clear()
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if bsp_results:
            save_partial_bsp_data(bsp_results, bsp_seg_ids, save_dir, file_idx * 1000 + len(v_data) // 200)
            logging.info(f"✅ 已保存文件 {file}")