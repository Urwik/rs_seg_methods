import sys

sys.path.append('/home/arvc/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.trainer import TrainerBase
from engines.trainers.tester import TesterBase
from models.pointnet import PointNetBinSeg
from engines.dataset.base import BaseDataset

# ----------------------------------------------
# TRAINERS
# ----------------------------------------------

class pnTrainer(TrainerBase):
    def __init__(self, config_node=None, build_dirs=True):
        super().__init__(config_node, build_dirs=build_dirs)
    
    @staticmethod
    def get_name():
        return 'PointNetBinSeg'

    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        self.config['dataset']['fixed_size'] = True
        dataset = BaseDataset(config_node=self.config)

        return dataset

    def build_model(self):

        self.console.debug("BUILDING MODEL")

        self.model = PointNetBinSeg(n_feat=len(self.config['dataset']['feat_idx']),
                                    device=self.device,
                                    ).to(self.device)

        self.config['train']['model'] = self.model.get_name()

        return self.model


# ----------------------------------------------
# TESTERS
# ----------------------------------------------

class pnTester(TesterBase):
    
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

        self.model = PointNetBinSeg(n_feat=len(self.train_config['dataset']['feat_idx']),
                                    device=self.device,
                                    ).to(self.device)

        return self.model