import sys
import warnings
import requests
import yaml
import os
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.pnTrainer import pnTrainer as Trainer
# from engines.trainers.pn2Trainer import pn2Trainer as Trainer
# from engines.trainers.mkTrainer import mkTrainer as Trainer
# from engines.trainers.ptV3Trainer import ptV3Trainer as Trainer
# from engines.trainers.Trainer import TrainerBase

def compute_optim_threshold(model_dir : str):
    
    CONFIG_FILE_PATH = os.path.join(model_dir, 'config.yaml')
    
    with open(CONFIG_FILE_PATH, 'r') as file:
        train_config = yaml.safe_load(file)
    
    train_config['dataset']['mode'] = 'train'
    
    if 'train_dir' not in train_config['dataset'] and 'root_dir' in train_config['dataset']:
        train_config['dataset']['train_dir'] = train_config['dataset']['root_dir']
    
    trainer = Trainer(config_node=train_config, build_dirs=False)
    trainer.set_output_dir(model_dir)
    trainer.set_model_weights( os.path.join(model_dir, 'model.pth'))
    trainer.valid_one_epoch()
    trainer.save_threshold()  


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    MODEL_EXPERIMENTS_DIR = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNetBinSeg'    

    for model_dir in os.listdir(MODEL_EXPERIMENTS_DIR):
        model_dir = os.path.join(MODEL_EXPERIMENTS_DIR, model_dir)
        if os.path.isdir(model_dir):
            if not os.path.exists(os.path.join(model_dir, 'threshold.npy')):
                try:
                    compute_optim_threshold(model_dir)
                
                except Exception as e:
                    print(f"Model directory: {model_dir} error occurred while computing threshold...")
                    print(f"Error: {e}")