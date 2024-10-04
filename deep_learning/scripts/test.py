import sys
import warnings
import requests
import yaml
sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

# from engines.trainers.pnTrainer import pnTester as Tester
# from engines.trainers.pn2Trainer import pn2Tester as Tester
from engines.trainers.mkTrainer import mkTester as Tester
# from engines.trainers.ptV3Trainer import ptV3Trainer
# from engines.trainers.Trainer import TrainerBase

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    with open('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/test.yaml', 'r') as file:
        test_config = yaml.safe_load(file)

    test_config['enable_metrics'] = False
    test_config['device'] = 'cuda:1'
    test_config['save_infered_clouds'] = True
    test_config['model_dir'] = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/MinkUNet34C/240628061431'
    test_config['dataset_dir'] = '/home/arvc/Fran/datasets/complex_structure/orto/ply_xyzln'
    test_config['output_dir'] = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/visualization/results/MinkUNet34C/orto_pruebas_no_validas'
    tester = Tester(test_config_node= test_config)
    
    tester.test()

    print("Sending notification...")
    header = f'Test Completed: {tester.model.__class__.__name__}'
    message = f'Precision: {tester.test_metrics.precision()}\nRecall: {tester.test_metrics.recall()}\nF1 Score: {tester.test_metrics.f1_score()}\nmIoU: {tester.test_metrics.mIou()}'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })