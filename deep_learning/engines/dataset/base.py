import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import yaml


ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_END = "\033[0m"


class BaseDataset(Dataset):
    def __init__(self, config_node=None):
        super().__init__()

        self.config = None
        self.mode = 'train'
        self.root_dir = None
        self.label_idx = 3
        self.coord_idx = [0,1,2]
        self.feat_idx = [0,1,2]
        self.sequences = [0,1,2,3,4,5,6,7,9]
        self.fixed_size = False
        self.normalize = False
        self.sorted = False
        self.force_binary_labels = True
        self.add_range_feature = False
        self.enable_weights = False
        self.dataset = []
        self.weights = []

        # INPUT IS A YAML NODE
        if isinstance(config_node, dict):
            self.config = config_node
            
            self.mode = self.config['dataset']['mode']
            if self.mode == "train":
                self.root_dir = self.config['dataset']['train_dir']
            elif self.mode == "test":
                self.root_dir = self.config['dataset']['test_dir']
            else:
                print(f"{ANSI_RED}DATASET MODE NOT SPECIFIED{ANSI_END}")
                exit()

            try:
                self.label_idx = self.config['dataset']['label_idx']
            except:
                pass
            try:
                self.coord_idx = self.config['dataset']['coord_idx']
            except:
                pass
            try:
                self.feat_idx = self.config['dataset']['feat_idx']
            except:
                pass
            try:
                self.sequences = self.config['dataset']['sequences']
            except:
                pass
            try:
                self.fixed_size = self.config['dataset']['fixed_size']
            except:
                pass
            try:
                self.normalize = self.config['dataset']['normalize']
            except:
                pass
            try:
                self.force_binary_labels = self.config['dataset']['force_binary_labels']
            except:
                pass
            try:
                self.add_range_feature = self.config['dataset']['add_range_feature']
            except:
                pass
            try:
                self.enable_weights = self.config['dataset']['compute_weights']
            except:
                pass
            

        # If config path is provided, load the config file
        elif isinstance(config_node, str):
            print(f"{ANSI_GREEN}GETTING CONFIG INFO FROM: {ANSI_END}{config_node}")
            if os.path.exists(config_node):
                with open(config_node) as file:
                    self.config = yaml.safe_load(file)

                self.mode = self.config['dataset']['mode']
                if self.mode == "train":
                    self.root_dir = self.config['dataset']['train_dir']
                elif self.mode == "test":
                    self.root_dir = self.config['dataset']['test_dir']
                else:
                    print(f"{ANSI_RED}DATASET MODE NOT SPECIFIED{ANSI_END}")
                    exit()
                    
            try:
                self.label_idx = self.config['dataset']['label_idx']
            except:
                pass
            try:
                self.coord_idx = self.config['dataset']['coord_idx']
            except:
                pass
            try:
                self.feat_idx = self.config['dataset']['feat_idx']
            except:
                pass
            try:
                self.sequences = self.config['dataset']['sequences']
            except:
                pass
            try:
                self.fixed_size = self.config['dataset']['fixed_size']
            except:
                pass
            try:
                self.normalize = self.config['dataset']['normalize']
            except:
                pass
            try:
                self.force_binary_labels = self.config['dataset']['force_binary_labels']
            except:
                pass
            try:
                self.add_range_feature = self.config['dataset']['add_range_feature']
            except:
                pass
            try:
                self.enable_weights = self.config['dataset']['compute_weights']
            except:
                pass
            try: 
                self.sorted = self.config['dataset']['sorted']
            except:
                pass
                
            else:
                print(f"{ANSI_YELLOW} !! PATH TO FILE DO NOT EXIST: {ANSI_END}{config_node}")
                print(f"{ANSI_RED} EXITING... {ANSI_END}")

        self.get_dataset()



    def get_dataset(self):

        for root_seq in os.listdir(self.root_dir):
            tmp_path = os.path.join(self.root_dir, root_seq)
            if os.path.isdir(tmp_path):
                if int(root_seq) in self.sequences:
                    
                    if self.fixed_size:
                        sufix = "ply_xyzln_fixedSize"
                    else:
                        sufix = "ply_xyzln"
                        
                    clouds_path = os.path.join(self.root_dir, root_seq, sufix)
                    print(f"{ANSI_YELLOW}\tSEQ PATH: {clouds_path}{ANSI_END}")
                    for file in os.listdir(clouds_path):
                        if file.endswith(".ply"):
                            self.dataset.append(os.path.join(clouds_path, file))

            elif os.path.isfile(tmp_path):
                if tmp_path.endswith(".ply"):
                    self.dataset.append(tmp_path)
                     
        if self.enable_weights:
            self.compute_weights()


    def compute_weights(self):
        # COMPUTE WEIGHTS FOR EACH LABEL IN THE WHOLE DATASET
        print("-" * 50)
        print("COMPUTING LABEL WEIGHTS")
        for file in tqdm(self.dataset):
            # READ THE FILE
            path_to_file = os.path.join(self.root_dir, file)
            ply = PlyData.read(path_to_file)
            data = ply["vertex"].data
            data = np.array(list(map(list, data)))

            # CONVERT TO BINARY LABELS
            labels = data[:, self.label_idx].copy()

            if self.binary:
                labels[labels > 0] = 1

            labels = np.sort(labels, axis=None)
            k_lbl, weights = np.unique(labels, return_counts=True)
            # SI SOLO EXISTE UNA CLASE EN LA NUBE (SOLO SUELO)
            if k_lbl.size < 2:
                if k_lbl[0] == 0:
                    weights = np.array([1, 0])
                else:
                    weights = np.array([0, 1])
            else:
                weights = weights / len(labels)

            if len(self.weights) == 0:
                self.weights = weights
            else:
                self.weights = np.vstack((self.weights, weights))

        self.weights = np.mean(self.weights, axis=0).astype(np.float32)
    
    def sort_by_distance(self, data):
        coords = data[:, self.coord_idx].copy()
        labels = data[:, self.label_idx].copy()
        features = data[:, self.feat_idx].copy()
        distance = np.sqrt(np.sum(coords ** 2, axis=-1))
        idx = np.argsort(distance)
        return coords[idx], features[idx], labels[idx]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        coords = data[:, self.coord_idx].copy()
        features = data[:, self.feat_idx].copy()
        labels = data[:, self.label_idx].copy()

        if self.sorted:
            coords, features, labels = self.sort_by_distance(data)

        if self.feat_idx[:3] == [0, 1, 2]:
            if self.normalize:
                xyz = coords.copy()
                centroid = np.mean(xyz, axis=0)
                xyz -= centroid
                furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
                xyz /= furthest_distance
                features[:, [0, 1, 2]] = xyz

        if self.add_range_feature:
            xyz = coords.copy()
            range = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            range = range[:, None]
            features = np.hstack((features, range))

        if self.force_binary_labels:
            labels[labels > 0] = 1

        return coords, features, labels
        

class TrainDataset(BaseDataset):
    pass


class TestDataset(BaseDataset):
    def __init__(self, train_config, test_config):
        super().__init__(config_node=train_config)

        self.train_config = train_config
        self.root_dir = '/home/arvc/Fran/datasets/retTruss_Test'
        self.coord_idx = self.train_config['dataset']['coord_idx']
        self.feat_idx = self.train_config['dataset']['feat_idx']
        self.label_idx = self.train_config['dataset']['label_idx']
        self.fixed_size = self.train_config['dataset']['fixed_size']
        self.normalize = self.train_config['dataset']['normalize']
        self.force_binary_labels = self.train_config['dataset']['force_binary_labels']
        self.add_range_feature = self.train_config['dataset']['add_range_feature']
        self.compute_weights = self.train_config['dataset']['compute_weights']
        self.sequences = test_config['sequences']

        self.get_dataset()
        
if __name__ == "__main__":

    dataset = BaseDataset(
        config_path="/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/config.yaml"
    )

    # dataset = BASE(
    #     mode="train",
    #     train_dir="/home/arvc/datasets/retTruss/test_code/ply_xyzL",
    #     coord_idx=[0, 1, 2],
    #     feat_idx=[0, 1, 2],
    #     label_idx=3,
    #     normalize=False,
    #     force_binary_labels=True,
    #     add_range_feature=False,
    #     compute_weights=False,
    # )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=None,
        drop_last=True
    )

    for batch, data in enumerate(loader):
        coords = data[0]
        features = data[1]
        labels = data[2]


        has_coords_nan = torch.isnan(coords).any()
        has_features_nan = torch.isnan(features).any()
        has_labels_nan = torch.isnan(labels).any()

        print(f'Has coords nan: {has_coords_nan}')
        print(f'Has features nan: {has_features_nan}')
        print(f'Has labels nan: {has_labels_nan}')


        # print(f'Data shape: {len(data)}')
        # print(f'cloud {batch}: {data[0][0].shape}')

        # # PATHS
        # print(data[3])

        # COORDS
        # print(data[0][0])

        # FEATURES
        # print(data[1][0])

        # LABELS
        # print(data[2][0])

        print("-" * 10)
