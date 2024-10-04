import requests

import os
import yaml
import time
import warnings
import datetime

import sys
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

# from engines.trainers.pnTrainer import pnTrainer
# from engines.trainers.pn2Trainer import pn2Trainer
# from engines.trainers.mkTrainer import mkTrainer
from engines.utils.csv_parser import csvExperimentQueue
# from scripts.ptV3Trainer import ptV3Trainer

def create_experiments_csv(path='/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/exp_queue.csv'):

    features_list = [[0,1,2], [4,5,6], [0,1,2,4,5,6], [0,1,2,7], [7]]
    scheduler_list = ['cosine', 'plateau']
    optimizer_list = ['adam', 'sgd']
    termination_criterion_list = ['loss', 'precision', 'miou']
    voxel_size_list = [0.01, 0.05, 0.1, 0.15]
    grid_size_list = [0.01, 0.05, 0.1, 0.15]
    batch_size_list = [8, 16, 32]
    threshold_method_list = ['pr', 'roc']
    

    # config_base_path = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
    # if os.path.exists(config_base_path):
    #     with open(config_base_path) as file:
    #         config_node = yaml.safe_load(file)
            
    exp_queue_csv = csvExperimentQueue(path)
    exp_queue_csv.createCSV()
    
    exp_config_list = []

    # for termination_criterion in termination_criterion_list:
    # for threshold_method in threshold_method_list:
    for feature in features_list:
        # for voxel_size in voxel_size_list:
        # current_config = [feature, threshold_method, termination_criterion]
        current_config = [feature]
        
        exp_queue_csv.appendData(current_config)

    # for voxel_size in voxel_size_list:
    #     config = [voxel_size, [0,1,2], 'pr', 'loss']
    #     exp_queue_csv.appendData(config)

if __name__ == '__main__':
    
    OUTPUT_FILE_PATH = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNetBinSeg/exp_queue.csv'
    
    create_experiments_csv(OUTPUT_FILE_PATH)
    
    print('Experiments CSV created')