"""
retTuss dataset

Author: FJ SOLER (f.soler@umh.es)
"""

import os
import numpy as np
from plyfile import PlyData
import yaml
from copy import deepcopy

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class retTrussDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        
        self.sequences = [0,1,2,3,4,5,6,7,8,9]
        self.coord_idx = [0, 1, 2]
        self.label_idx = [3]
        self.force_binary_labels = True
        
        self.exp_cfg = kwargs.pop('exp_cfg')
        self.feat_idx = self.exp_cfg['feat_idx']
        self.normalize = self.exp_cfg['normalize']
        
        print(f"FEAT IDX: {self.feat_idx}")
        print(f"NORMALIZE: {self.normalize}")
        
        # CONFIG_FILE_PATH = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
        # with open(CONFIG_FILE_PATH, 'r') as file:
        #     train_config = yaml.safe_load(file)
            
        # self.cfg = train_config
        # self._parse_config()
        
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)
            

    def get_data_list(self):
        
        data_list = []
        if os.path.basename(self.data_root) == "ply_xyzln":
            for file in os.listdir(self.data_root):
                if file.endswith(".ply"):
                    data_list.append(os.path.join(self.data_root, file))
        else: 
            for root_seq in os.listdir(self.data_root):
                tmp_path = os.path.join(self.data_root, root_seq)
                if os.path.isdir(tmp_path):
                    if int(root_seq) in self.sequences:
                        
                        sufix = "ply_xyzln"
                            
                        clouds_path = os.path.join(self.data_root, root_seq, sufix)
                        print(f"\tSEQ PATH: {clouds_path}")
                        for file in os.listdir(clouds_path):
                            if file.endswith(".ply"):
                                data_list.append(os.path.join(clouds_path, file))
        
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        
        ply = PlyData.read(data_path)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))
        
        coords = data[:, self.coord_idx].copy()
        features = data[:, self.feat_idx].copy()
        labels = data[:, self.label_idx].copy()
        labels = labels.astype(np.int64)
        if len(labels.shape) > 1:
            labels= labels.flatten()
        
        if self.feat_idx[:3] == [0, 1, 2]:
            if self.normalize:
                xyz = coords.copy()
                centroid = np.mean(xyz, axis=0)
                xyz -= centroid
                furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
                xyz /= furthest_distance
                features[:, [0, 1, 2]] = xyz

        # if self.add_range_feature:
        #     xyz = coords.copy()
        #     range = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
        #     range = range[:, None]
        #     features = np.hstack((features, range))

        if self.force_binary_labels:
            labels[labels > 0] = 1
        
        data_dict = dict(
            coord=coords,
            feat=features,
            segment=labels,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    # def prepare_train_data(self, idx):
    #     # load data
    #     data_dict = self.get_data(idx)
    #     data_dict = self.transform(data_dict)
    #     return data_dict

    # def prepare_test_data(self, idx):
    #     # load data
    #     data_dict = self.get_data(idx)
    #     data_dict = self.transform(data_dict)
    #     result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        
    #     if "origin_segment" in data_dict:
    #         assert "inverse" in data_dict
    #         result_dict["origin_segment"] = data_dict.pop("origin_segment")
    #         result_dict["inverse"] = data_dict.pop("inverse")

    #     data_dict_list = []
    #     for aug in self.aug_transform:
    #         data_dict_list.append(aug(deepcopy(data_dict)))

    #     fragment_list = []
    #     for data in data_dict_list:
    #         if self.test_voxelize is not None:
    #             data_part_list = self.test_voxelize(data)
    #         else:
    #             data["index"] = np.arange(data["coord"].shape[0])
    #             data_part_list = [data]
    #         for data_part in data_part_list:
    #             if self.test_crop is not None:
    #                 data_part = self.test_crop(data_part)
    #             else:
    #                 data_part = [data_part]
    #             fragment_list += data_part

    #     for i in range(len(fragment_list)):
    #         fragment_list[i] = self.post_transform(fragment_list[i])
    #     result_dict["fragment_list"] = fragment_list
    #     return result_dict


    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: 0,  # "ground"
            1: 1,  # "truss"
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 0,  # "ground"
            1: 1,  # "ground"
        }
        return learning_map_inv
