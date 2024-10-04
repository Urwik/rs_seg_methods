import requests

import os
import yaml
import time
import warnings
import datetime
import ast

import sys
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.pnTrainer import pnTrainer
from engines.trainers.pn2Trainer import pn2Trainer
from engines.trainers.mkTrainer import mkTrainer
from engines.utils.csv_parser import csvExperimentQueue
# from scripts.ptV3Trainer import ptV3Trainer


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    config_base_path = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
    if os.path.exists(config_base_path):
        with open(config_base_path) as file:
            config_node = yaml.safe_load(file)
                    
                    
    EXP_QUEUE_FILE = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/MinkUNet34C/exp_list.csv'

                    
    exp_queue = csvExperimentQueue(EXP_QUEUE_FILE)
    
    while exp_queue.get_num_rows_in_csv() > 0:
        
        # now = datetime.datetime.now()
        # start_time = now.replace(hour=15, minute=30, second=0, microsecond=0)  # 15:30 
        # end_time = now.replace(hour=6, minute=0, second=0, microsecond=0)  # 06:00
        
        # if now <= end_time:
        #     end_time = end_time
        # else:
        #     end_time = end_time + datetime.timedelta(days=1)

        # # Now check if the current time is between 15:30 and 06:00
        # if now >= start_time or now <= end_time:
            # print("Valid time, running experiment...")
        
        first_row = exp_queue.getFirstRow()
        
        voxel_size = float(first_row[0])
        feature = ast.literal_eval(first_row[1])
        threshold_method = str(first_row[2])
        termination_criterion = str(first_row[3])
        
        config_node['train']['termination_criterion'] = termination_criterion
        config_node['train']['threshold_method'] = threshold_method
        config_node['dataset']['feat_idx'] = feature
        config_node['train']['voxel_size'] = voxel_size
        config_node['train']['loader']['num_workers'] = 10
            
        # trainer = pnTrainer(config_node)
        # trainer = pn2Trainer(config_node)
        trainer = mkTrainer(config_node)
        # trainer.csv_train.set_output_file('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/experiments/train.csv')
        trainer.train()
        
        exp_queue.removeRowByIndex(0)
        
        
        
        try:
            print("Sending notification...")
            header = f'Training Completed: {trainer.model.__class__.__name__}'
            message = f'Remain experiments: {exp_queue.get_num_rows_in_csv()}\nPrecision: {trainer.best_model_metrics.precision()}\nRecall: {trainer.best_model_metrics.recall()}\nF1 Score: {trainer.best_model_metrics.f1_score()}\nmIoU: {trainer.best_model_metrics.mIou()}'
            requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })
        except:
            print("Error sending notification...")
        # else:
        #     print("Sleeping...")
        #     time.sleep(3600)
            
    
    print("No more experiments to run...")
    print("Sending notification...")
    header = f'Queue finished: {trainer.model.__class__.__name__}'
    message = f'All experiments have been completed'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })