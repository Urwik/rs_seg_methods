import os
import sys
import warnings
import requests
import yaml
from tqdm import tqdm
import pandas as pd
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.utils.csv_parser import csvTestStruct

# from engines.trainers.pnTrainer import pnTester as Tester
from engines.trainers.pn2Trainer import pn2Tester as Tester
# from engines.trainers.mkTrainer import mkTester as Tester
# from engines.trainers.ptV3Trainer import ptV3Trainer as Tester

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    MODEL = 'PointNet2BinSeg'
    DATASET_SEQS = ["orto", "crossed", "00", "01", "02", "03"]



    with open('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/test.yaml', 'r') as file:
        test_config = yaml.safe_load(file)

    models_dir = os.listdir(f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/{MODEL}')
    
    for seq in DATASET_SEQS:
        DATASET_DIR = f'/home/arvc/Fran/datasets/complex_structure/{seq}/ply_xyzln'
        test_config['dataset_dir'] = DATASET_DIR
        for _, exp in enumerate(tqdm(models_dir)):
            tmp_path = f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/{MODEL}/{exp}'
            if os.path.isdir(tmp_path):
                
                TEST_DF = csvTestStruct()
                if int(exp) in TEST_DF.df['EXPERIMENT_ID'].values:
                    TEST_DF.filter('EXPERIMENT_ID', int(exp))
                    if seq in TEST_DF.df['DATASET_DIR'].values:
                        tqdm.write(f'Experiment {exp} and sequence {seq} already tested')
                        continue

                tqdm.write("\n")
                tqdm.write(f'#'*50)
                tqdm.write(f'TESTING: {Tester.__name__} - {exp} - {seq}')
                test_config['model_dir'] = tmp_path
                test_config['device'] = 'cuda:1'
                test_config['save_infered_clouds'] = False
                tester = Tester(test_config_node= test_config)
                tester.test()
                tqdm.write(tester.test_metrics.get_as_string())

    print("Sending notification...")
    header = f'Testing Completed'
    message = f'Precision: {tester.test_metrics.precision()}\nRecall: {tester.test_metrics.recall()}\nF1 Score: {tester.test_metrics.f1_score()}\nmIoU: {tester.test_metrics.mIou()}'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })