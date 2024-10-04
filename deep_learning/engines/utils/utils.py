from plyfile import PlyData, PlyElement
import numpy as np
import os

def NptoPly(data_array, out_dir, ply_name, features, binary):

    if ply_name[-4:] == '.ply':
        abs_file_path = os.path.join(out_dir, ply_name)
    else:
        abs_file_path = os.path.join(out_dir, ply_name + '.ply')

    cloud = list(map(tuple, data_array))
    vertex = np.array(cloud, dtype=features)
    el = PlyElement.describe(vertex, 'vertex')
    if binary:
        PlyData([el]).write(abs_file_path)
    else:
        PlyData([el], text=True).write(abs_file_path)


def save_pred_as_ply(coords, label_pred, out_dir_, filename_):
    """
    coords : np.array of shape (N, 3)
    label_pred : np.array of shape (N,) 
    """
    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]
    
    if coords.ndim != 2:
        coords = coords.squeeze(axis=0)
    
    if label_pred.ndim == 1:
        label_pred = label_pred[:,None]
    
    cloud = np.hstack((coords, label_pred))

    # coords = coords[:,None]
    # label_pred = label_pred[:,None]
    
    NptoPly(cloud, out_dir_, filename_, features=feat_xyzlabel, binary=True)