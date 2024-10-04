import requests

import os
import yaml
import time
import warnings
import datetime
import ast

import sys
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.pnTrainer import pnTrainer as Trainer
# from engines.trainers.pn2Trainer import pn2Trainer as Trainer
# from engines.trainers.mkTrainer import mkTrainer as Trainer
# from scripts.ptV3Trainer import ptV3Trainer as Trainer
from engines.utils.csv_parser import csvExperimentQueue


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    config_base_path = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
    if os.path.exists(config_base_path):
        with open(config_base_path) as file:
            config_node = yaml.safe_load(file)
                    
    EXP_QUEUE_FILE = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNetBinSeg/exp_queue.csv'

                    
    exp_queue = csvExperimentQueue(EXP_QUEUE_FILE)
    
    norm = False
    
    while exp_queue.get_num_rows_in_csv() > 0:
        
        first_row = exp_queue.getFirstRow()
        
        feature = ast.literal_eval(first_row[0])
        # threshold_method = str(first_row[1])
        # termination_criterion = str(first_row[2])
        
        # config_node['train']['termination_criterion'] = termination_criterion
        # config_node['train']['threshold_method'] = threshold_method
        config_node['dataset']['feat_idx'] = feature
        config_node['train']['loader']['num_workers'] = 10
        config_node['train']['device'] = 'cuda:1'
        config_node['train']['loader']['batch_size'] = 12
        config_node['dataset']['normalize'] = False
            
        trainer = Trainer(config_node)
        trainer.train()
        
        exp_queue.removeRowByIndex(0)
        
        try:
            print("Sending notification...")
            header = f'Training Completed: {trainer.model.__class__.__name__}'
            message = f'Remain experiments: {exp_queue.get_num_rows_in_csv()}\nPrecision: {trainer.best_model_metrics.precision()}\nRecall: {trainer.best_model_metrics.recall()}\nF1 Score: {trainer.best_model_metrics.f1_score()}\nmIoU: {trainer.best_model_metrics.mIou()}'
            requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })
        except:
            print("Error sending notification...")
            
    
    print("No more experiments to run...")
    print("Sending notification...")
    header = f'Queue finished: {trainer.model.__class__.__name__}'
    message = f'All experiments have been completed'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })