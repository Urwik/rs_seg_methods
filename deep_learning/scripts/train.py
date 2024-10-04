import sys
import warnings
import requests
import yaml
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.pnTrainer import pnTrainer as Trainer
# from engines.trainers.pn2Trainer import pn2Trainer as Trainer
# from engines.trainers.mkTrainer import mkTrainer as Trainer
# from engines.trainers.ptV3Trainer import ptV3Trainer as Trainer

if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    
    CONFIG_FILE_PATH = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/train.yaml'
    
    with open(CONFIG_FILE_PATH, 'r') as file:
        train_config = yaml.safe_load(file)
    
    train_config['dataset']['device'] = 'cuda:1'
    trainer = Trainer(config_node=train_config)
    
    # trainer.set_num_workers(2)
    # trainer.set_batch_size(4)
    # trainer.build_dataloaders()
    trainer.train()

    print("Sending notification...")
    header = f'Training Completed: {trainer.config["train"]["model"]}'
    message = f'Precision: {trainer.best_model_metrics.precision()}\nRecall: {trainer.best_model_metrics.recall()}\nF1 Score: {trainer.best_model_metrics.f1_score()}\nmIoU: {trainer.best_model_metrics.mIou()}'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })