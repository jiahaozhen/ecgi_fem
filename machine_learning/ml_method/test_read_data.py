import os
import numpy as np


def load_dataset_generator(data_dir, batch_size=1000):
    files = sorted(os.listdir(data_dir))
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        batch_data = []
        batch_labels = []
        for f in batch_files:
            # 假设每个文件是.npy格式，且包含'X'和'y'两个key
            data = np.load(os.path.join(data_dir, f), allow_pickle=True)
            batch_data.append(data['X'])
            batch_labels.append(data['y'])
        yield np.array(batch_data), np.array(batch_labels)


if __name__ == '__main__':
    data_dir = 'machine_learning/data/dataset/d300_standard_dataset'
    batch_size = 1000
    for X_batch, y_batch in load_dataset_generator(data_dir, batch_size):
        print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
        # 这里可以对每个batch进行训练或处理
