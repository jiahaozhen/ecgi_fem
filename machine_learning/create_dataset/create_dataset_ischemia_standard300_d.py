# 将d转化为标准化数据（电压差）
import os
import numpy as np
import logging
import h5py
from utils.signal_processing_tools import transfer_bsp_to_standard300lead


def save_partial_bsp_data(standard_d_results, seg_ids, save_dir, partial_idx):
    standard_d_array = np.array(standard_d_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(
        save_dir, f"ischemia_d_standard_part_{partial_idx:03d}.h5"
    )
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=standard_d_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset_dl/d300_standard_dataset'
    d_data_dir = 'machine_learning/data/dataset_dl/d_dataset'

    # save_dir = 'machine_learning/data/dataset_ml/d300_standard_dataset'
    # d_data_dir = 'machine_learning/data/dataset_ml/d_dataset'

    d_data_files = [
        os.path.join(d_data_dir, f) for f in os.listdir(d_data_dir) if f.endswith('.h5')
    ]
    d_data_files.sort()

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_data_files):
        with h5py.File(file, "r") as data:
            d_data = data['X'][:]
            seg_ids = data['y'][:]

        standard_d_results = []
        standard_d_seg_ids = []

        for i, (d, seg_id) in enumerate(zip(d_data, seg_ids)):
            try:
                standard_d = transfer_bsp_to_standard300lead(d)
                standard_d_results.append(standard_d)
                standard_d_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if standard_d_results:
            save_partial_bsp_data(
                standard_d_results, standard_d_seg_ids, save_dir, file_idx
            )
            logging.info(f"✅ 已处理文件 {file}")
