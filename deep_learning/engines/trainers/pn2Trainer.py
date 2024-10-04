import sys
import warnings
import math
import requests
import torch 
from torch.utils.data import DataLoader

sys.path.append('/home/arvc/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.trainer import TrainerBase
from engines.trainers.tester import TesterBase
from models.pointnet2 import PointNet2BinSeg
from engines.dataset.base import BaseDataset


# ----------------------------------------------
# TRAINERS
# ----------------------------------------------
class pn2Trainer(TrainerBase):
    def __init__(self, config_node=None):
        super().__init__(config_node)

    @staticmethod
    def get_name():
        return 'PointNet2BinSeg'

    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        self.config['dataset']['fixed_size'] = True
        dataset = BaseDataset(config_node=self.config)

        return dataset

    def build_model(self):

        self.console.debug("BUILDING MODEL")

        self.model = PointNet2BinSeg(n_feat=len(self.config['dataset']['feat_idx']),
                                         device=self.device,
                                         dropout_=True
                                         ).to(self.device)
        
        self.config['train']['model'] = self.model.get_name()


        return self.model


    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
        self.model_input = (self.coords, self.features)
            

# ----------------------------------------------
# TESTERS
# ----------------------------------------------
class pn2Tester(TesterBase):
    
    def build_dataset(self):
        self.console.debug("BUILDING DATASET")
        
        self.train_config['dataset']['mode'] = 'test'
        self.train_config['dataset']['fixed_size'] = True
        self.train_config['dataset']['test_dir'] = self.test_config['dataset_dir']
        self.train_config['dataset']['sequences'] = self.test_config['sequences']

        self.test_dataset = BaseDataset(config_node=self.train_config)

        return self.test_dataset

    def build_model(self):
        self.console.debug("BUILDING MODEL")
        
        self.model = PointNet2BinSeg(n_feat=len(self.train_config['dataset']['feat_idx']),
                                         device=self.device,
                                         dropout_=True
                                         ).to(self.device)
        
        return self.model

    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
        self.model_input = (self.coords, self.features)