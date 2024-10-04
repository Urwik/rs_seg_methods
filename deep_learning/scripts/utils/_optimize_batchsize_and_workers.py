import sys
import warnings
import requests
import yaml
import multiprocessing as mp
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

# from engines.trainers.pnTrainer import pnTrainer as Trainer
# from engines.trainers.pn2Trainer import pn2Trainer as Trainer
# from engines.trainers.mkTrainer import mkTrainer as Trainer
from engines.trainers.ptV3Trainer import ptV3Trainer as Trainer

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    results = {}
    
    TRAIN_CFG_FILE_PATH = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
    TEST_CFG_FILE_PATH = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/test.yaml'
    
    with open(TRAIN_CFG_FILE_PATH, 'r') as file:
        train_config = yaml.safe_load(file)
    
    with open(TEST_CFG_FILE_PATH, 'r') as file:
        test_config = yaml.safe_load(file)
    
    
    for num_workers in range(0, mp.cpu_count(), 2):  
    
        trainer = Trainer(config_node=train_config)
        
        trainer.set_num_workers(num_workers)
        trainer.set_batch_size(4)
        trainer.build_dataloaders()
        duration = trainer.train_X_batch(10)
        
        print(f"Num workers: {num_workers}, Duration: {duration}")
        results[num_workers] = duration

    for num_workers, duration in results.items():
        print(f"Num workers: {num_workers}, Duration: {duration}")