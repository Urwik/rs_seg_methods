import sys
import warnings
import math
import requests
import torch 
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import os
import open3d as o3d
import numpy as np

sys.path.append('/home/arvc/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.trainer import TrainerBase
from engines.trainers.tester import TesterBase   
from engines.dataset.base import TestDataset, BaseDataset
from engines.utils.utils import save_pred_as_ply
from models.minkunet import MinkUNet34C

class mkTrainer(TrainerBase):
    def __init__(self, config_node=None, build_dirs=True):
        super().__init__(config_node, build_dirs=build_dirs)
        
        
    @staticmethod
    def get_name():
        return 'MinkUNet34C'

    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        self.config['dataset']['fixed_size'] = False
        dataset = BaseDataset(config_node=self.config)

        return dataset

    def build_model(self):

        self.console.debug("BUILDING MODEL")
        torch.cuda.device(self.device)

        self.model = MinkUNet34C(in_channels=len(self.config['dataset']['feat_idx']),
                                    device=self.device
                                    ).to(self.device)

        self.config['train']['model'] = self.model.get_name()

        return self.model


    def build_dataloaders(self):

        self.console.debug("BUILDING DATALOADERS")

        # INSTANCE DATASET
        train_dataset = self.build_dataset()

        # SPLIT VALIDATION AND TRAIN
        train_size = math.floor(len(train_dataset) * self.config['train']['train_valid_split'])
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(74))


        self.train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=self.config['train']['loader']['batch_size'],
                                        num_workers=self.config['train']['loader']['num_workers'],
                                        shuffle=self.config['train']['loader']['shuffle'],
                                        pin_memory=self.config['train']['loader']['pin_memory'],
                                        drop_last=self.config['train']['loader']['drop_last'],
                                        collate_fn=ME.utils.batch_sparse_collate)

        self.valid_dataloader = DataLoader(dataset=valid_dataset,
                                        batch_size=self.config['train']['loader']['batch_size'],
                                        num_workers=self.config['train']['loader']['num_workers'],
                                        shuffle=self.config['train']['loader']['shuffle'],
                                        pin_memory=self.config['train']['loader']['pin_memory'],
                                        drop_last=self.config['train']['loader']['drop_last'],
                                        collate_fn=ME.utils.batch_sparse_collate)

        
        return self.train_dataloader, self.valid_dataloader

    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
            
        self.coords[:,1:] /= self.config['train']['voxel_size']

        self.mk_in_field = ME.TensorField(
            features= self.features,
            coordinates= self.coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device)
            
        self.model_input = self.mk_in_field.sparse()

    def forward_model(self):
        self.model_output = self.model(self.model_input)
        propagated_out = self.model_output.slice(self.mk_in_field)
        self.model_output = propagated_out.F.squeeze()



class mkTester(TesterBase):
    def __init__(self, test_config_node):
        super().__init__(test_config_node)

    def build_dataset(self):
        self.console.debug("BUILDING DATASET")
        self.train_config['dataset']['mode'] = 'test'
        self.train_config['dataset']['test_dir'] = self.test_config['dataset_dir']
        self.train_config['dataset']['sequences'] = self.test_config['sequences']
        
        self.test_dataset = BaseDataset(config_node=self.train_config)

        return self.test_dataset

    def build_model(self):

        self.console.debug("BUILDING MODEL")
        torch.cuda.device(self.device)

        self.model = MinkUNet34C(in_channels=len(self.train_config['dataset']['feat_idx']),
                                device=self.device
                                ).to(self.device)

        return self.model

    def build_dataloader(self):

        self.console.debug("BUILDING DATALOADERS")

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                        batch_size=1,
                                        num_workers=self.num_workers,
                                        shuffle=False,  
                                        pin_memory=True,
                                        drop_last=False,
                                        collate_fn=ME.utils.batch_sparse_collate)
        
        return self.test_dataloader

    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
        
        self.voxelized_coords = self.coords.clone()
        self.voxelized_coords[:,1:] /= self.train_config['train']['voxel_size']
        # self.coords[:,1:] /= self.train_config['train']['voxel_size']
        
        # self.coords = self.coords.squeeze(0)
        # self.features = self.features.squeeze(0)
        # self.labels = self.labels.squeeze(0)
            

        self.mk_in_field = ME.TensorField(
            features= self.features,
            coordinates= self.voxelized_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device)
            
        self.model_input = self.mk_in_field.sparse()

    def forward_model(self):
        self.model_output = self.model(self.model_input)
        propagated_out = self.model_output.slice(self.mk_in_field)
        self.model_output = propagated_out.F.squeeze()

    def save_cloud(self):
        cloud_path = self.test_dataset.dataset[self.current_batch_num]
        cloud_name = os.path.basename(cloud_path)
        
        o3d_cloud = o3d.io.read_point_cloud(cloud_path)
        coords = np.asarray(o3d_cloud.points)
        
        # if torch.is_tensor(self.coords):
        #     coords = self.coords.detach().cpu().numpy().copy()
        # else:
        #     coords = self.coords.copy()
            
        # coords = coords[:, 1:]
        
        if torch.is_tensor(self.pred_fix):
            pred = self.pred_fix.detach().cpu().numpy().copy()
        else:
            pred = self.pred_fix.copy()
        
        save_pred_as_ply(coords, pred, self.output_dir, cloud_name)