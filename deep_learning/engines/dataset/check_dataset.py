import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from plyfile import PlyData



dataset_path = '/home/arvc/Fran/datasets/retTruss/'
variant = 'ply_xyzln'

dataset = []


def check_dataset():
    global dataset_path
    global dataset
    global variant
    for root_seq in os.listdir(dataset_path):
        seq_path = os.path.join(dataset_path, root_seq)

        if os.path.isdir(seq_path):
            clouds_path = os.path.join(seq_path, variant)

            for file in os.listdir(clouds_path):
                if file.endswith(".ply"):
                    dataset.append(os.path.join(clouds_path, file))

    for cloud in tqdm(dataset):
        ply = PlyData.read(cloud)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))
        # print(f"Points shape {points.shape}")

        for point in data:
            if np.isnan(point).any():
                print(f"Found nan in {cloud}")
                exit()

if __name__ == '__main__':
    check_dataset()