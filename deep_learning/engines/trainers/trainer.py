import os
import yaml
import numpy as np
import copy
import torch 
from torch.utils.data import DataLoader, Subset
import sklearn.metrics as metrics
from datetime import datetime
import shutil
import math
import time
import MinkowskiEngine as ME
from models import PointNetBinSeg, PointNet2BinSeg, MinkUNet34C
from engines.dataset.base import BaseDataset
from engines.utils.metrics import Metrics
from engines.utils.console import *
from engines.utils.csv_parser import csvTrainStruct, csvTestStruct
from engines.losses.bce_lovasz import BCELovaszLoss
from engines.losses.lovaszbinary import LovaszLoss

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_END = "\033[0m"

class TrainerBase():

    def __init__(self, config_node=None, build_dirs=True):

        self.config_file_abs_path = config_node
        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.model_input = None
        self.loss = 1
        self.lr = 0.01
        self.epochs = 100
        self.batch_size = 16
        self.threshold_method = 'pr'
        self.improve_patiente = 5
        
        
        self.metrics = Metrics()
        self.best_model_metrics = Metrics()
        
        self.console = Console(enable=False)
        self.csv_train = csvTrainStruct()
        # self.csv_test = csvTestStruct()
        
        # If config path is provided, load the config file
        if isinstance(config_node, str):
            
            self.console.info(f"GETTING CONFIG INFO FROM: {config_node}", ANSI_CYAN)
            self.config_file_abs_path = config_node
            with open(config_node, 'r') as file:
                self.config = yaml.safe_load(file)

            self.console.enable = self.config['train']['debug']
                
            self.build_device()
            self.build_model()
            if build_dirs: self.build_directories()
            if build_dirs: self.save_config_file()
            self.build_loss_fn()
            self.build_dataloaders()
            self.build_optimizer()
            self.build_scheduler()
            self.build_initial_model_quantifier()
            self.epochs = self.config['train']['epochs']
            self.batch_size = self.config['train']['loader']['batch_size']
            self.threshold_method = self.config['train']['threshold_method']
            self.config['train']['termination_criterion']
        
        elif isinstance(config_node, dict):
            self.console.info(f"GETTING CONFIG INFO FROM CONFIG NODE", ANSI_CYAN)

            self.config = config_node
            
            self.console.enable = self.config['train']['debug']
            
            self.build_device()
            self.build_model()
            if build_dirs: self.build_directories()
            if build_dirs: self.save_config_file()
            self.build_loss_fn()
            self.build_dataloaders()
            self.build_optimizer()
            self.build_scheduler()
            self.build_initial_model_quantifier()
            self.epochs = self.config['train']['epochs']
            self.batch_size = self.config['train']['loader']['batch_size']
            self.threshold_method = self.config['train']['threshold_method']
            self.config['train']['termination_criterion']
        
        else:
            self.console.info(f"ERROR OCCURED WHILE PASSING ARGUMENT TO THE CONSTRUCTOR", ANSI_RED)

    # -----------------------------------------------------
    # BUILDERS
    # -----------------------------------------------------  
            
    def build_directories(self):

        self.console.debug("BUILDING DIRECTORIES")

        folder_name = datetime.today().strftime('%y%m%d%H%M%S')
        self.csv_train.EXPERIMENT_ID = folder_name

        self.out_dir = os.path.join(self.config['train']['output_dir'], self.model.get_name(), folder_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        return self.out_dir
    
    def build_device(self):

        self.console.debug("BUILDING DEVICE")
        
        if torch.cuda.is_available():
            self.device = torch.device(self.config['train']['device'])
        else:
            self.console.info("CUDA DEVICE NOT AVAILABLE", ANSI_RED)
            self.device = torch.device("cpu")

        torch.cuda.set_device(self.device) # set the device to use    

        self.config['train']['device'] = str(self.device)
        self.console.info(f"DEVICE: {self.device}", ANSI_CYAN)

        self.csv_train.DEVICE = self.device

        return self.device
    
    def build_model(self):

        self.console.debug("BUILDING MODEL")

        if self.config["train"]["model"] == "pointnet":
            self.model = PointNetBinSeg(n_feat=len(self.config['dataset']['feat_idx']),
                                        device=self.device,
                                        ).to(self.device)

        elif self.config["train"]["model"] == "pointnet2":
            self.model = PointNet2BinSeg(n_feat=len(self.config['dataset']['feat_idx']),
                                         device=self.device,
                                         dropout_=True
                                         ).to(self.device)

        elif self.config["train"]["model"] == "minkunet34c":
            self.model = MinkUNet34C(in_channels=len(self.config['dataset']['feat_idx']),
                                     device=self.device
                                     ).to(self.device)
        else:
            print("MODEL NOT FOUND")

        self.config['train']['model'] = self.model.get_name()

        return self.model

    def build_loss_fn(self):

        self.console.debug("BUILDING LOSS FUNCTION")

        if self.config['train']['loss_fn'] == "bce":
            self.loss_fn = torch.nn.BCELoss()
        elif self.config['train']['loss_fn'] == "bcew":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config['train']['loss_fn'] == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.config['train']['loss_fn'] == "bce_lovasz":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config['train']['loss_fn'] == "lovasz":
            self.loss_fn = LovaszLoss()
        else:
            self.console.info("LOSS FUNCTION NOT FOUND", ANSI_RED)

        return self.loss_fn

    def build_dataset(self):
        self.console.debug("BUILDING DATASET")

        dataset = BaseDataset(config_node=self.config)

        return dataset
    
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
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=self.config['train']['loader']['batch_size'],
                                        num_workers=self.config['train']['loader']['num_workers'],
                                        shuffle=self.config['train']['loader']['shuffle'],
                                        pin_memory=self.config['train']['loader']['pin_memory'],
                                        drop_last=self.config['train']['loader']['drop_last'])

        self.valid_dataloader = DataLoader(dataset=valid_dataset,
                                        batch_size=self.config['train']['loader']['batch_size'],
                                        num_workers=self.config['train']['loader']['num_workers'],
                                        shuffle=self.config['train']['loader']['shuffle'],
                                        pin_memory=self.config['train']['loader']['pin_memory'],
                                        drop_last=self.config['train']['loader']['drop_last'])
        
        return self.train_dataloader, self.valid_dataloader

    def build_optimizer(self):

        self.console.debug("BUILDING OPTIMIZER")

        self.lr = self.config['train']['init_lr']
        
        if self.config['train']['optimizer'] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.config['train']['optimizer'] == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.config['train']['optimizer'] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            self.console.info("OPTIMIZER NOT FOUND", ANSI_RED)
            
        return self.optimizer

    def build_scheduler(self):

        if self.config['train']['scheduler'] == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1, verbose=False)

        elif self.config['train']['scheduler'] == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif self.config['train']['scheduler'] == "onecycle":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, total_steps=1000, epochs=10, steps_per_epoch=100, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1, verbose=False)
        else:   
            self.lr_scheduler = None

        return self.lr_scheduler

    def build_initial_model_quantifier(self):
        
        self.improve_patiente = self.config['train']['improve_patience']
        
        if self.config['train']['termination_criterion'] == "loss":
            self.model_quantifier = 1
        else:
            self.model_quantifier = 0
    
    def build_model_input(self):
        self.coords = self.data[0].to(self.device, dtype=torch.float32)
        self.features = self.data[1].to(self.device, dtype=torch.float32)
        self.labels = self.data[2].to(self.device, dtype=torch.float32)
        self.model_input = self.features
    
    def build_prediction(self):
        m = torch.nn.Sigmoid()
        self.prediction = m(self.model_output)

    # -----------------------------------------------------
    # SETTERS
    # -----------------------------------------------------

    def set_num_workers(self, num_workers):
        self.config['train']['loader']['num_workers'] = num_workers
        
    def set_batch_size(self, batch_size):
        self.config['train']['loader']['batch_size'] = batch_size

    def set_model_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
   
    def set_output_dir(self, output_dir):
        self.out_dir = output_dir
    
    # -----------------------------------------------------
    # SAVE FUNCTIONS
    # -----------------------------------------------------
    def save_config_file(self):

        self.console.debug("SAVING CONFIG FILE")
        target_path = os.path.join(self.out_dir, 'config.yaml')

        # shutil.copyfile(self.config_file_abs_path, target_path)
        
        with open(target_path, 'w') as file:
            yaml.safe_dump(self.config, file, default_flow_style=False)
        # with open(target_path, 'w') as file:
        #     file.write(yaml.serialize(self.config))


        return target_path

    def save_model(self):
        torch.save(self.model.state_dict(), self.out_dir + f'/model.pth')
    
    def save_metrics(self, duration):
        with open(self.out_dir + '/metrics.txt', 'a') as file:
            file.write('-'*30)
            file.write(f'\nEPOCH: {self.current_epoch}\n')
            file.write('-'*30)
            file.write(f'\n  Precision: {self.metrics.precision()}\n')
            file.write(f'  Recall:    {self.metrics.recall()}\n')
            file.write(f'  F1 Score:  {self.metrics.f1_score()}\n')
            file.write(f'  Accuracy:  {self.metrics.accuracy()}\n')
            file.write(f'  mIoU:      {self.metrics.mIou()}\n')
            file.write(f'  Duration:  {duration}\n')

    def save_total_training_time(self, duration):
        with open(self.out_dir + '/metrics.txt', 'a') as file:
            file.write('-'*30)
            file.write(f'\nTOTAL TRAINING TIME: {duration}')
    
    def save_threshold(self):
        # write threshold to numpy file
        self.console.debug(f"SAVING THRESHOLD VALUE: {self.threshold}")
        np.save(self.out_dir + '/threshold.npy', self.threshold)

    # -----------------------------------------------------
    # TRAIN UTILS
    # -----------------------------------------------------
    def forward_model(self):
        self.model_output = self.model(self.model_input)
    
    def step(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def compute_loss(self):
        if self.loss.__class__.__name__ == 'BCELovaszLoss':
            self.loss = self.loss_fn(self.model_output, self.labels)
        if self.loss.__class__.__name__ == 'LovaszLoss':
            self.loss = self.loss_fn(self.model_output, self.labels)
        else:
            self.loss = self.loss_fn(self.prediction, self.labels)

    def compute_valid_loss(self):
        self.valid_loss = self.loss_fn(self.prediction, self.labels)

    def model_improved(self):
        if self.config['train']['termination_criterion'] == "loss":
            if self.valid_loss < self.model_quantifier:
                self.save_model()
                self.model_quantifier = self.valid_loss
                self.reset_improve_patience()
                return True
            else:
                self.improve_patiente -= 1
                return False

        
        elif self.config['train']['termination_criterion'] == "precision":
            difference = self.metrics.precision() - self.model_quantifier
            if difference > 0: # 0.005 is the minimum improvement to consider
                self.save_model()
                self.model_quantifier = self.metrics.precision()
                if difference > 0.005:
                    self.reset_improve_patience()
                return True
            else:
                self.improve_patiente -= 1
                return False
                
        elif self.config['train']['termination_criterion'] == "f1_score":
            difference = self.metrics.f1_score() - self.model_quantifier
            if difference > 0: # 0.005 is the minimum improvement to consider
                self.save_model()
                self.model_quantifier = self.metrics.f1_score()
                if difference > 0.005:
                    self.reset_improve_patience()
                return True
            else:
                self.improve_patiente -= 1
                return False
                
        elif self.config['train']['termination_criterion'] == "miou":
            difference = self.metrics.mIou() - self.model_quantifier
            if difference > 0: # 0.005 is the minimum improvement to consider
                self.save_model()
                self.model_quantifier = self.metrics.mIou()
                if difference > 0.005:
                    self.reset_improve_patience()
                return True
            else:
                self.improve_patiente -= 1
                return False

    def update_saved_model(self):
        if self.model_improved():
            self.save_model()
            self.save_threshold()
            print('-'*50)
            precision_ = self.metrics.precision()
            print(f'{ANSI_GREEN} NEW MODEL SAVED {ANSI_END} WITH PRECISION: {ANSI_GREEN}{precision_:.4f}{ANSI_END}')
            return True
        else:
            return False
    
    def update_learning_rate(self):
        if self.config['train']['scheduler'] == "step":
            self.lr_scheduler.step()

        elif self.config['train']['scheduler'] == "plateau":
            self.lr_scheduler.step(self.valid_loss)

    def update_threshold(self, pred, label):
        
        self.console.debug("Computing best threshold value:")
        method = self.config['train']['threshold_method']
        
        if method == "roc":
            fpr, tpr, thresholds = metrics.roc_curve(label, pred)
            gmeans = np.sqrt(tpr * (1 - fpr))
            index = np.argmax(gmeans)
            self.threshold = thresholds[index]

        elif method == "pr":
            precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
            f1_score = (2 * precision * recall) / (precision + recall)
            f1_score = np.nan_to_num(f1_score)
            index = np.argmax(f1_score)
            self.threshold = thresholds[index]

        elif method == "tuning":
            thresholds = np.arange(0.0, 1.0, 0.0001)
            f1_score = np.zeros(shape=(len(thresholds)))
            for index, tmp_thrshld in enumerate(thresholds):
                prediction_ = np.where(pred > tmp_thrshld, 1, 0).astype(int)
                f1_score[index] = metrics.f1_score(label, prediction_)

            index = np.argmax(f1_score)
            self.threshold = thresholds[index]
        else:
            print('Error in the name of the method to use for compute best threshold')
        
        self.console.debug(f"\tThreshold Value: {self.threshold}", ANSI_CYAN)

        return self.threshold
    
    def apply_pred_threshold(self, pred, threshold):
        # pred = pred.cpu().numpy()
        return np.where(pred > threshold, 1, 0).astype(int)    

    def reset_improve_patience(self):
        self.improve_patiente = self.config['train']['improve_patience']

    def termination_criterion(self):
        if self.improve_patiente == 0:
            return True
        else:
            return False        

    def before_step(self):
        pass

    def after_train_step(self):
        if self.current_batch_num  % 10 == 0 or self.current_batch_num == 0:  # print every (% X) batches
            print(f' - [Batch: {self.current_batch_num}/{self.train_dataloader.__len__()}]'
                    f' [Train Loss: {self.loss:.4f}]') 

    def after_valid_step(self):
        if self.current_batch_num  % 10 == 0 or self.current_batch_num == 0:  # print every (% X) batches
            print(f' - [Batch: {self.current_batch_num}/{self.valid_dataloader.__len__()}]'
                    f' [Valid Loss: {self.valid_loss:.4f}]')

    def compute_valid_metrics(self):
        self.console.debug("COMPUTING VALIDATION METRICS...")
        
        if torch.is_tensor(self.prediction):
            self.prediction = self.prediction.detach().cpu().numpy()
        if torch.is_tensor(self.labels):
            self.labels = self.labels.detach().cpu().numpy() 
        
        if np.ndim(self.prediction) > 1:
            self.prediction = self.prediction.flatten()
        if np.ndim(self.labels.shape) > 1:
            self.labels = self.labels.flatten()
            
        if self.prediction.max() == self.prediction.min():
            print(f'\033[93mWARNING!! ALL VALUES IN PREDICTION ARE: {self.prediction.max()}\033[0m')
            
        if self.labels.max() == self.labels.min():
            print(f'\033[93mWARNING!! ALL VALUES IN LABELS ARE: {self.labels.max()}\033[0m')
        
        self.threshold = self.update_threshold(self.prediction, self.labels)
        
        self.pred_fix = self.apply_pred_threshold(self.prediction, self.threshold)
            
        self.metrics.set_metrics(self.pred_fix, self.labels)
    
    
    def print_metrics(self):
        self.metrics.print()        

    def clear_gpu_memory(self):
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            time.sleep(1)
    
    def count_parameters_and_memory(self):
        total_params = 0
        total_memory = 0  # in bytes

        for param in self.model.parameters():
            # Number of parameters in the current layer
            num_params = param.numel()
            
            # Size in bytes per element (typically 4 bytes for float32, 2 bytes for float16)
            memory_per_param = param.element_size()

            # Total parameters
            total_params += num_params

            # Total memory in bytes
            total_memory += num_params * memory_per_param

        return total_params, total_memory

    # -----------------------------------------------------
    # TRAIN FUNCTIONS
    # -----------------------------------------------------

    def train_X_batch(self, num_batches):
        from time import time
        
        print('-' * 50)
        print('TRAINING ONLY ONE BATCH')
        print('-'*50)
        self.num_used_clouds = 0
        self.current_batch_num = 0

        self.model.train()
        
        start = time()
        batches_durations = np.empty(shape=(num_batches))
        for batch, data in enumerate(self.train_dataloader):
            self.data = data
            self.current_batch_num = batch

            self.build_model_input()

            self.forward_model()
            
            self.build_prediction()

            self.compute_loss()

            self.step()

            self.after_train_step()
            end = time()
            batches_durations[batch] = end - start
            
            if batch == num_batches - 1:
                break
        
        return np.mean(batches_durations)

    def train_one_epoch(self):
        print('-' * 50)
        print('TRAINING')
        print('-'*50)
        self.num_used_clouds = 0
        self.current_batch_num = 0

        self.model.train()
        for batch, data in enumerate(self.train_dataloader):
            self.data = data
            self.current_batch_num = batch

            self.build_model_input()

            self.forward_model()
            
            self.build_prediction()

            self.compute_loss()

            self.step()

            self.after_train_step()

    def valid_one_epoch(self):

        # VALIDATION
        print('-' * 50)
        print('VALIDATION')
        print('-'*50)
        self.num_used_clouds = 0
        self.current_batch_num = 0

        self.model.eval()
        
        pred_array = None
        label_array = None
        
        self.metrics = Metrics()

        with torch.no_grad():
            for batch, data in enumerate(self.valid_dataloader):
                self.data = data
                self.current_batch_num = batch

                self.build_model_input()

                self.forward_model()
                
                self.build_prediction()
                
                self.compute_valid_loss()

                self.after_valid_step()
                
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

            # ESTABLECER LA METRICAS MEDIAS DE VALIDACION
            
            self.prediction = pred_array
            self.labels = label_array
            self.compute_valid_metrics()

    def train(self):
        start_time = datetime.now()
        
        self.console.debug("PRINTING CONFIG", ANSI_CYAN)
        for key, value in self.config.items():
            for key, value in value.items():
                self.console.debug(f'\t{key}: {value}')
        
        for epoch in range(self.epochs):
            epoch_start_time = datetime.now()
            self.current_epoch = epoch
            
            print('-'*50); print(f'EPOCH:{ANSI_GREEN}{epoch}{ANSI_END} | LR:{ANSI_GREEN}{self.lr_scheduler.get_last_lr()[0]}{ANSI_END} | Patience:{ANSI_GREEN}{self.improve_patiente}{ANSI_END}'); print('-'*50)
            

            self.train_one_epoch()
            self.clear_gpu_memory()
            
            self.valid_one_epoch()
            self.clear_gpu_memory()

            self.print_metrics()

            if self.metrics.is_nan():
                self.console.info('ERROR: METRICS ARE NAN', ANSI_RED)
                input('Press Enter to continue...')
                
            
            if self.update_saved_model():
                self.csv_train.BEST_EPOCH = epoch
                self.csv_train.PRECISION = self.metrics.precision()
                self.csv_train.RECALL = self.metrics.recall()
                self.csv_train.F1_SCORE = self.metrics.f1_score()
                self.csv_train.MIOU = self.metrics.mIou()
                self.csv_train.ACCURACY = self.metrics.accuracy()
                self.csv_train.TP = self.metrics.tp
                self.csv_train.FP = self.metrics.fp
                self.csv_train.TN = self.metrics.tn
                self.csv_train.FN = self.metrics.fn
                self.best_model_metrics = copy.deepcopy(self.metrics)

            if self.termination_criterion():
                self.console.info('TERMINATION CRITERION REACHED', ANSI_CYAN)
                break

            self.update_learning_rate()

            epoch_end_time = datetime.now()
            epoch_duration = epoch_end_time - epoch_start_time

            self.save_metrics(epoch_duration)
            print('-'*50); print(f' EPOCH DURATION: {ANSI_YELLOW} {epoch_duration}{ANSI_END}'); print('-'*50); print('\n')

            self.csv_train.MODEL = self.model.get_name()
            self.csv_train.NUM_EPOCHS = epoch
            self.csv_train.SEQUENCES = self.config['dataset']['sequences']
            self.csv_train.FEATURES = self.config['dataset']['feat_idx']
            self.csv_train.NORMALIZATION = self.config['dataset']['normalize']
            self.csv_train.OPTIMIZER = self.config['train']['optimizer']
            self.csv_train.SCHEDULER = self.config['train']['scheduler']
            self.csv_train.THRESHOLD_METHOD = self.config['train']['threshold_method']
            self.csv_train.TERMINATION_CRITERIA = self.config['train']['termination_criterion']
            self.csv_train.VOXEL_SIZE = self.config['train']['voxel_size']

        end_time = datetime.now()
        training_duration = end_time - start_time
        
        self.save_total_training_time(training_duration)

        self.csv_train.DURATION = training_duration
        self.csv_train.append()

        self.console.info('TRAINING COMPLETED', ANSI_GREEN)
        self.console.info(f'Duration: {training_duration}')
        self.console.info(f'METRICS:')
        self.best_model_metrics.print()

    