import os
import numpy as np
import logging
import h5py
from utils.signal_processing_tools import project_bsp_on_surface, load_bsp_pts


def save_partial_bsp_data(surface_d_results, seg_ids, save_dir, partial_idx):
    surface_d_array = np.array(surface_d_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(
        save_dir, f"ischemia_d_surface_part_{partial_idx:03d}.h5"
    )
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=surface_d_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh_file = 'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    save_dir = 'machine_learning/data/dataset_dl/d_surface_dataset'
    d_standard_data_dir = 'machine_learning/data/dataset_dl/d_standard_dataset'

    # save_dir = 'machine_learning/data/dataset_ml/d_surface_dataset'
    # d_standard_data_dir = 'machine_learning/data/dataset_ml/d_standard_dataset'

    d_standard_data_files = [
        os.path.join(d_standard_data_dir, f)
        for f in os.listdir(d_standard_data_dir)
        if f.endswith('.h5')
    ]
    d_standard_data_files.sort()

    original_pts = load_bsp_pts()

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_standard_data_files):
        with h5py.File(file, "r") as data:
            d_standard_data = data['X'][:]
            seg_ids = data['y'][:]

        surface_d_results = []
        surface_d_seg_ids = []

        for i, (d, seg_id) in enumerate(zip(d_standard_data, seg_ids)):
            try:
                surface_d = project_bsp_on_surface(d)
                surface_d_results.append(surface_d)
                surface_d_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if surface_d_results:
            save_partial_bsp_data(
                surface_d_results, surface_d_seg_ids, save_dir, file_idx
            )
            logging.info(f"✅ 已处理文件 {file}")
