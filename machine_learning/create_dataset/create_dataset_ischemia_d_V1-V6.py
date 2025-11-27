import os
import numpy as np
import logging
from utils.helper_function import transfer_bsp_to_standard12lead

def save_partial_bsp_data(d_V1_V6_results, seg_ids, save_dir, partial_idx):
    d_V1_V6_array = np.array(d_V1_V6_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"ischemia_d_V1_V6_part_{partial_idx:03d}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(partial_file, X=d_V1_V6_array, y=seg_ids_array)
    logging.info(f"✅ 已保存 {partial_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset/d_V1_V6_dataset'

    # Load v_data and seg_ids from ischemia dataset
    d_data_dir = 'machine_learning/data/dataset/d_dataset'
    d_data_files = [os.path.join(d_data_dir, f) for f in os.listdir(d_data_dir) if f.endswith('.npz')]
    d_data_files.sort()  # Ensure consistent order
    d_data = []
    seg_ids = []

    lead_index=np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_data_files):
        with np.load(file) as data:
            d_data = data['X']
            seg_ids = data['y']

        d_V1_V6_results = []
        d_V1_V6_seg_ids = []

        for i, (d, seg_id) in enumerate(zip(d_data, seg_ids)):
            try:
                d_V1_V6 = transfer_bsp_to_standard12lead(d, lead_index=lead_index)[:, 3:9]
                d_V1_V6_results.append(d_V1_V6)
                d_V1_V6_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if d_V1_V6_results:
            save_partial_bsp_data(d_V1_V6_results, 
                                  d_V1_V6_seg_ids, 
                                  save_dir, file_idx)
            logging.info(f"✅ 已处理文件 {file}")