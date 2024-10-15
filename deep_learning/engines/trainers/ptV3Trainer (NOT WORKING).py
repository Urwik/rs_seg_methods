import sys
import warnings
import requests
import math
from addict import Dict
import torch
from functools import partial

sys.path.append('/home/arvc/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.trainer import TrainerBase
from engines.dataset.pointcept_dataset import PointCeptDataset
from models._pointTransfomerV3 import PointTransformerV3

from repos.Pointcept.pointcept.datasets.utils import point_collate_fn

# import serialization

class ptV3Trainer(TrainerBase):
    def __init__(self, config_node=None):
        super().__init__(config_node)

    @staticmethod
    def get_name():
        return 'PointTransformerV3'
    
    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        self.config['dataset']['fixed_size'] = False
        dataset = PointCeptDataset(config_node=self.config)

        return dataset
    
    def set_num_workers(self, num_workers):
        self.config['train']['loader']['num_workers'] = num_workers
    
    def set_batch_size(self, batch_size):
        self.config['train']['loader']['batch_size'] = batch_size
    
    def build_dataloaders(self):

        self.console.debug("BUILDING DATALOADERS")

        # INSTANCE DATASET
        train_dataset = self.build_dataset()

        # SPLIT VALIDATION AND TRAIN
        train_size = math.floor(len(train_dataset) * self.config['train']['train_valid_split'])
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(74))

        # INSTANCE DATALOADERS
        self.train_dataloader = torch.utils.data.DataLoader(
                                    dataset=train_dataset,
                                    batch_size=self.config['train']['loader']['batch_size'],
                                    num_workers=self.config['train']['loader']['num_workers'],
                                    shuffle=self.config['train']['loader']['shuffle'],
                                    pin_memory=self.config['train']['loader']['pin_memory'],
                                    drop_last=self.config['train']['loader']['drop_last'],
                                    collate_fn=partial(point_collate_fn))

        self.valid_dataloader = torch.utils.data.DataLoader(
                                    dataset=valid_dataset,
                                    batch_size=self.config['train']['loader']['batch_size'],
                                    num_workers=self.config['train']['loader']['num_workers'],
                                    shuffle=self.config['train']['loader']['shuffle'],
                                    pin_memory=self.config['train']['loader']['pin_memory'],
                                    drop_last=self.config['train']['loader']['drop_last'],
                                    collate_fn=partial(point_collate_fn))
    
        return self.train_dataloader, self.valid_dataloader

    def build_model(self):
        self.console.debug("BUILDING MODEL")

        self.model = PointTransformerV3(
            in_channels=len(self.config['dataset']['feat_idx']),
            # order=("z", "z-trans"),
            # stride=(2, 2, 2, 2),
            # enc_depths=(2, 2, 2, 6, 2),
            # enc_channels=(32, 64, 128, 256, 512),
            # enc_num_head=(2, 4, 8, 16, 32),
            # enc_patch_size=(48, 48, 48, 48, 48),
            # dec_depths=(2, 2, 2, 2),
            # dec_channels=(64, 64, 128, 256),
            # dec_num_head=(4, 4, 8, 16),
            # dec_patch_size=(48, 48, 48, 48),
            # mlp_ratio=4,
            # qkv_bias=True,
            # qk_scale=None,
            # attn_drop=0.0,
            # proj_drop=0.0,
            # drop_path=0.3,
            # pre_norm=True,
            # shuffle_orders=True,
            # enable_rpe=False,
            # enable_flash=True,
            # upcast_attention=False,
            # upcast_softmax=False,
            # cls_mode=False,
            # pdnorm_bn=False,
            # pdnorm_ln=False,
            # Si pdnorm_bn y pdnorm_ln son False, entonces todo los siguientes no se usan.
            # pdnorm_decouple=True,
            # pdnorm_adaptive=False,
            # pdnorm_affine=True,
            # pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        )

        self.config['train']['model'] = self.model.get_name()
        self.model = self.model.cuda()

        return self.model

    def build_model_input(self):
        for key in self.data.keys():
            if isinstance(self.data[key], torch.Tensor):
                self.data[key] = self.data[key].cuda(non_blocking=True)

        self.labels = self.data['segment'].to(self.device, dtype=torch.float32)

        self.model_input = self.data

    def forward_model(self):
        # with torch.cuda.amp.autocast(enabled=True):
        self.model_output = self.model(self.model_input)
        self.model_output = self.model_output.feat

    def build_prediction(self):
        m = torch.nn.Sigmoid()
        self.prediction = m(self.model_output)
        self.prediction = self.prediction.squeeze()


    # def train_one_epoch(self):
    #     print('-' * 50)
    #     print('TRAINING')
    #     print('-'*50)
    #     self.num_used_clouds = 0
    #     self.current_batch_num = 0      

    #     self.model.train()
    #     for batch, data in enumerate(self.train_dataloader):
    #         self.data = data
    #         self.current_batch_num = batch

    #         self.build_model_input()
            
    #         self.forward_model()
            
    #         self.build_prediction()

    #         self.compute_loss()

    #         self.step()

    #         self.after_train_step()