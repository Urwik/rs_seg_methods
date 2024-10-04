import os
import yaml
import numpy as np
import torch 
import sys
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import sklearn.metrics as metrics
from engines.dataset.base import TestDataset, BaseDataset
from engines.utils.metrics import Metrics
from engines.utils.console import *
from models import PointNetBinSeg, PointNet2BinSeg, MinkUNet34C
import time
from engines.utils.csv_parser import csvTestStruct

sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')
from tests.metrics_parser import MetricsParser
from engines.utils.utils import save_pred_as_ply

class TesterBase():
    def __init__(self, test_config_node = None):
        self.test_config = None
        self.test_model_dir = None
        self.dataset_dir = None
        self.num_workers = 0
        self.device = None 
        self.save_infered_clouds = False
        self.debug = False
        self.model = None
        self.train_config = None
        self.enable_metrics = True
        self.output_dir = ''
        self.test_metrics = Metrics()
        self.csv_test = csvTestStruct()
        self.console = Console()

        if test_config_node is not None:
            self.test_config = test_config_node
            self.test_model_dir = self.test_config['model_dir']
            self.dataset_dir = self.test_config['dataset_dir']
            self.device = self.test_config['device']
            try:
                self.save_infered_clouds = self.test_config['save_infered_clouds']
            except:
                pass
            try:
                self.output_dir = self.test_config['output_dir']
            except:
                pass
            try:
                self.enable_metrics = self.test_config['enable_metrics']
            except:
                pass
            try:
                self.console.enable = self.test_config['debug']
            except:
                pass
    

        self.get_train_config()
        self.build_device()
        self.build_model()
        self.build_dataset()
        self.build_dataloader()
        self.load_model()
        self.load_threshold()
        self.build_output_dir()
    # ----------------------------------------------
    # BUILDERS
    # ----------------------------------------------
    def build_model(self):
        pass
    
    def build_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(self.device)
        else:
            self.console.info("CUDA DEVICE NOT AVAILABLE", ANSI_RED)
            self.device = torch.device("cpu")

        torch.cuda.set_device(self.device)
        
        self.console.debug(f"DEVICE: {self.device}", ANSI_GREEN)
        
        return self.device
    
    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        self.train_config['dataset']['mode'] = 'test'
        self.train_config['dataset']['test_dir'] = self.test_config['dataset_dir']

        self.test_dataset = BaseDataset(config_node=self.train_config)

        return self.test_dataset
        
    def build_dataloader(self):
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                        batch_size=1,
                                        num_workers=self.num_workers,
                                        shuffle=False,
                                        pin_memory=True,
                                        drop_last=False)
    
    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
        self.model_input = self.features
    
    def build_prediction(self):
        m = torch.nn.Sigmoid()
        self.prediction = m(self.model_output)

    def build_output_dir(self):
        if self.save_infered_clouds:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
    # ----------------------------------------------
    # SETTERS
    # ----------------------------------------------

    def set_test_model_dir(self, test_model_dir):
        self.test_model_dir = test_model_dir

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers
    # ----------------------------------------------
    # GETTERS
    # ----------------------------------------------

    def get_train_config(self):
        config_file_path = os.path.join(self.test_model_dir, 'config.yaml')
        
        with open(config_file_path, 'r') as file:
            self.train_config = yaml.safe_load(file)

    def get_train_best_epoch(self):
        metrics_parser = MetricsParser()
        metrics_parser.set_experiment(self.test_model_dir)
        target_metric = self.train_config['train']['termination_criterion']  

        if target_metric == 'miou':
            target_metric = 'mIoU'
        elif target_metric == 'f1':
            target_metric = 'F1 Score'
        elif target_metric == 'precision':
            target_metric = 'Precision'
        elif target_metric == 'recall':
            target_metric = 'Recall'
        elif target_metric == 'accuracy':
            target_metric = 'Accuracy'
        elif target_metric == 'loss':
            target_metric = 'Precision'
        
        metrics_parser.set_target_metric(target_metric)
        metric, epoch = metrics_parser.get_best_metric_value()
        self.best_epoch = epoch
        return self.best_epoch 

    def get_mean_train_epoch_duration(self):
        metrics_parser = MetricsParser()
        metrics_parser.set_experiment(self.test_model_dir)
        mean_duration = metrics_parser.get_mean_epoch_duration()
        return mean_duration

    # ----------------------------------------------
    # TEST UTILS
    # ----------------------------------------------
    def forward_model(self):
        self.model_output = self.model(self.model_input)
    
    def load_model(self):
        if self.model is None:
            self.build_model()
        

        self.console.debug(f"LOADING MODEL FROM: {os.path.join(self.test_model_dir, 'model.pth')}")
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.test_model_dir, 'model.pth'), map_location=self.device))
            self.console.debug("MODEL LOADED CORRECTLY", ANSI_GREEN)
        except:
            self.console.info("ERROR WHILE LOADING THE MODEL", ANSI_RED)
            
    def load_threshold(self):          
        threshold_path = os.path.join(self.test_model_dir, 'threshold.npy')
        try: 
            self.threshold = np.load(threshold_path)
        except:
            self.threshold = 0.5
            self.console.info("ERROR WHILE LOADING THE THRESHOLD", ANSI_RED)
            self.console.info(f"\tSETTING DEFAULT THRESHOLD TO {self.threshold}", ANSI_YELLOW)

    def apply_pred_threshold(self, pred, threshold):
        if torch.is_tensor(pred):
            prediction = pred.detach().cpu().numpy()
        else:
            prediction = pred
            
        if np.ndim(prediction) > 1:
            prediction = prediction.flatten()
            
        self.pred_fix = np.where(prediction > threshold, 1, 0).astype(int)
        return self.pred_fix    
        
    def export_test_results_to_csv(self):
        self.csv_test.EXPERIMENT_ID = self.test_model_dir.split('/')[-1]
        self.csv_test.DEVICE = self.device
        self.csv_test.MODEL = self.train_config['train']['model']
        self.csv_test.SEQUENCES = self.train_config['dataset']['sequences']
        self.csv_test.FEATURES = self.train_config['dataset']['feat_idx']
        self.csv_test.NORMALIZATION = self.train_config['dataset']['normalize']
        self.csv_test.OPTIMIZER = self.train_config['train']['optimizer']
        self.csv_test.SCHEDULER = self.train_config['train']['scheduler']
        self.csv_test.THRESHOLD_METHOD = self.train_config['train']['threshold_method']
        self.csv_test.TERMINATION_CRITERIA = self.train_config['train']['termination_criterion']
        self.csv_test.TEST_DATASET = self.test_config['dataset_dir']
        
        try:
            self.csv_test.VOXEL_SIZE = self.train_config['train']['voxel_size']
        except:
            pass
        
        try:    
            self.csv_test.GRID_SIZE = self.train_config['train']['grid_size']
        except:
            pass
        self.csv_test.BEST_EPOCH = self.get_train_best_epoch()
        self.csv_test.EPOCH_DURATION = self.get_mean_train_epoch_duration()
        # self.csv_test.INFERENCE_DURATION = 
        self.csv_test.PRECISION = self.test_metrics.precision()
        self.csv_test.RECALL = self.test_metrics.recall()
        self.csv_test.F1_SCORE = self.test_metrics.f1_score()
        self.csv_test.ACCURACY = self.test_metrics.accuracy()
        self.csv_test.MIOU = self.test_metrics.mIou()
        self.csv_test.TP = self.test_metrics.tp
        self.csv_test.FP = self.test_metrics.fp
        self.csv_test.TN = self.test_metrics.tn
        self.csv_test.FN = self.test_metrics.fn

        self.csv_test.append()

    def compute_metrics(self):
        if torch.is_tensor(self.prediction):
            prediction = self.prediction.detach().cpu().numpy()
        else:
            prediction = self.prediction
            
        if torch.is_tensor(self.labels):
            labels = self.labels.detach().cpu().numpy()
        else:
            labels = self.labels 
        
        if np.ndim(prediction) > 1:
            prediction = prediction.flatten()
        if np.ndim(labels.shape) > 1:
            labels = labels.flatten()
            
        if prediction.max() == prediction.min():
            current_cloud = os.path.basename(self.test_dataset.dataset[self.current_batch_num])
            self.console.debug(f"WARN: ALL VALUES IN PREDICTION ARE: {prediction.max()} - IN: {current_cloud}", ANSI_YELLOW)
            
        if labels.max() == labels.min():
            current_cloud = os.path.basename(self.test_dataset.dataset[self.current_batch_num])
            self.console.debug(f"WARN: ALL VALUES IN LABELS ARE: {labels.max()} - IN: {current_cloud}", ANSI_YELLOW)


        self.pred_fix = self.apply_pred_threshold(prediction, self.threshold)
            
        self.test_metrics.set_metrics(self.pred_fix, labels)
        
        if self.test_metrics.tp == 0 or self.test_metrics.fp == 0 or self.test_metrics.fn == 0 or self.test_metrics.tn == 0:
            self.console.debug(f'WARN: ZERO VALUES IN CONF. MATRIX: TN: {self.test_metrics.tn}, FP: {self.test_metrics.fp}, FN: {self.test_metrics.fn}, TP: {self.test_metrics.tp}', ANSI_YELLOW)

    def visualize(self):
        cloud = o3d.io.read_point_cloud(self.test_dataset.dataset[self.current_batch_num])
        
        colors = np.zeros((np.asarray(cloud.points).shape[0], 3))
        colors[self.pred_fix == 0] = [1, 0, 0]
        colors[self.pred_fix == 1] = [0, 1, 0]
        
        cloud.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([cloud])
    
    def save_cloud(self):
        cloud_name = os.path.basename(self.test_dataset.dataset[self.current_batch_num])
        
        if torch.is_tensor(self.coords):
            coords = self.coords.detach().cpu().numpy().copy()
        else:
            coords = self.coords.copy()
        
        if torch.is_tensor(self.pred_fix):
            label_pred = self.pred_fix.detach().cpu().numpy().copy()
        else:
            label_pred = self.pred_fix.copy()
        
        if label_pred.ndim == 1:
            label_pred = label_pred[:,None]
        
        save_pred_as_ply(coords, label_pred, self.output_dir, cloud_name)
    # ----------------------------------------------
    # TEST
    # ----------------------------------------------

    def test(self):
        if self.test_model_dir is None:
            self.console.info("TEST MODEL DIR NOT SET, PLEASE SET IT BEFORE TEST", ANSI_RED)
            exit()
        
        # self.get_train_config()
        # self.build_device()
        # self.build_model()
        # self.build_dataset()
        # self.build_dataloader()
        # self.load_model()
        # self.load_threshold()
        # self.build_output_dir()
        self.model.eval()
        
        pred_array = None
        label_array = None

        inference_start = time.time()

        precision, recall, f1_score, accuracy, mIou = [], [], [], [], []
        
        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                self.data = data
                self.current_batch_num = batch

                self.build_model_input()

                self.forward_model()
                
                self.build_prediction()

                if self.test_config['visualize']:
                    self.apply_pred_threshold(self.prediction, self.threshold)
                    self.visualize()

                if self.save_infered_clouds:
                    self.apply_pred_threshold(self.prediction, self.threshold)
                    self.save_cloud()

                # self.compute_metrics()

                # precision.append(self.test_metrics.precision())
                # recall.append(self.test_metrics.recall())
                # f1_score.append(self.test_metrics.f1_score())
                # accuracy.append(self.test_metrics.accuracy())
                # mIou.append(self.test_metrics.mIou())

                prediction = self.prediction.detach().cpu().numpy()
                label = self.labels.detach().cpu().numpy()
                
                if len(prediction.shape) > 1:
                    prediction = prediction.flatten()
                
                if len(label.shape) > 1:
                    label = label.flatten()

                if pred_array is None:
                    pred_array = prediction
                else:
                    pred_array = np.concatenate((pred_array, prediction), axis=0)
                    
                if label_array is None:
                    label_array = label
                else:
                    label_array = np.concatenate((label_array, label), axis=0)

            inference_end = time.time()
            inference_duration = inference_end - inference_start
            cloud_inference_duration = inference_duration / len(self.test_dataset)
            self.csv_test.INFERENCE_DURATION = cloud_inference_duration
            # ESTABLECER LA METRICAS MEDIAS DE VALIDACION
            self.prediction = pred_array
            self.labels = label_array
            
            if self.enable_metrics:
                self.compute_metrics()
                self.export_test_results_to_csv()