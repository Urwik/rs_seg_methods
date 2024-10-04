import numpy as np
import open3d as o3d

if __name__ == "__main__":
    # Load point cloud
    cloud = o3d.io.read_point_cloud("/home/arvc/Fran/datasets/complex_structure/00/ply_xyzln/00000.ply")
    labels = np.load("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept/exp/retTruss/semseg-pt-v3m1-0-base/result/00_00000_pred.npy")

    colors = np.zeros((len(cloud.points), 3))
    colors[labels == 0] = [0.1, 0.1, 0.1]     # Map 0 to black
    colors[labels == 1] = [0, 1, 0]  

    cloud.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([cloud])
