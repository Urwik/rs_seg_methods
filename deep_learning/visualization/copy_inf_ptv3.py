import torch
import torch.distributed as dist
import datetime
import os
import sys
import numpy as np
import open3d as o3d

sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')
from engines.utils.utils import save_pred_as_ply


clouds_map = {
    "orto":     ["00147", "00360", "00524", "00227", "00345", "00181"],
    "crossed":  ["00147", "00360", "00524", "00227", "00345", "00181"],
    "00": ["00000", "00001", "00002", "00003", "00004", "00005"],
    "01": ["00000", "00001", "00002", "00003", "00004", "00005"],
    "02": ["00000", "00001", "00002", "00003", "00004"],
    "03": ["00000", "00001", "00002", "00003", "00004", "00005"]
}



def get_coords(set, cloud_name):
    cloud_path = f"/home/arvc/Fran/datasets/complex_structure/{set}/ply_xyzln/{cloud_name}.ply"
    cloud = o3d.io.read_point_cloud(cloud_path)
    coords = np.asarray(cloud.points)
    return coords


def get_labels(exp_name, set, cloud_name):
    label_path = f"/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept/exp/retTruss/{exp_name}/{set_name}/result/{set_name}_{cloud_name}_pred.npy"
    labels = np.load(label_path)
    return labels


def save_cloud(labels, coords, output_dir, cloud_name):
    
    save_pred_as_ply(coords, labels, output_dir, cloud_name)

def export_to_results(exp_name, set_name):

    destination_path = f"/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/visualization/results/PointTransformerV3/{exp_name}/{set_name}"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for cloud_name in clouds_map[set_name]:
        labels = get_labels(exp_name, set_name, cloud_name)
        coords = get_coords(set_name, cloud_name)
        save_cloud(labels, coords, destination_path, cloud_name)


if __name__ == '__main__':
    
    EXP_NAMES = ["c", "nxnynz", "xyz", "xyzc", "xyznxnynz"]
    DATASETS = clouds_map.keys()

    for exp_name in EXP_NAMES:
        for set_name in DATASETS:
            export_to_results(exp_name, set_name)
    
    # start = "13:28:19,831"
    # time_start = datetime.datetime.strptime(start, "%H:%M:%S,%f")

    # end = "13:42:33,695"
    # time_end = datetime.datetime.strptime(end, "%H:%M:%S,%f")

    # duration = time_end - time_start
    # duration_in_hours = duration.total_seconds() / 3600
    # print(duration_in_hours)
