from datetime import datetime
import os
import time
import sys
sys.path.append("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept")
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)

from pointcept.engines.test import TESTERS

sys.path.append("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation")
from engines.utils.csv_parser import csvTestStruct



def test(features, cfg_path):
    # datasets = ["00", "01", "02", "03"]
    datasets = ["orto", "crossed"]
    
    # datasets = ["00"]
    
    
    for dataset in datasets:
        # args = default_argument_parser().parse_args()
        os.makedirs(f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept/exp/retTruss/{features}/{dataset}', exist_ok=True)
        cfg = default_config_parser(cfg_path, options=None)
        cfg = default_setup(cfg)
        cfg.test_root = f'/home/arvc/Fran/datasets/complex_structure/{dataset}/ply_xyzln'
        cfg.data['test']['data_root'] = cfg.test_root
        cfg.save_path = f'exp/retTruss/{features}/{dataset}'
        cfg.weight = f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept/exp/retTruss/{features}/model/model_best.pth'
        
        csv_test = csvTestStruct()
        csv_test.EXPERIMENT_ID = datetime.today().strftime('%y%m%d%H%M%S')
        csv_test.MODEL = 'PointTrasnformerV3'
        csv_test.FEATURES = features.upper()
        csv_test.GRID_SIZE = cfg.data['train']['transform'][0]['grid_size']
        csv_test.TEST_DATASET = dataset
        csv_test.OPTIMIZER = cfg.optimizer['type']
        csv_test.THRESHOLD_METHOD = "None"
        csv_test.TERMINATION_CRITERIA = "epochs"
        csv_test.SCHEDULER = cfg.scheduler['type']
        csv_test.NORMALIZATION = False
        # csv_test.DEVICE = cfg.device
        csv_test.SEQUENCES = [0,1,2,3,4,5,6,7,8,9]
        csv_test.BEST_EPOCH = 10
        
        
        
        inference_start = time.time()
        
        tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
        tester.test()
        
        inference_end = time.time()
        inference_duration = inference_end - inference_start
        cloud_inference_duration = inference_duration / len(tester.test_loader.dataset)
        csv_test.INFERENCE_DURATION = cloud_inference_duration
        csv_test.PRECISION = tester.metrics.precision()
        csv_test.RECALL = tester.metrics.recall()
        csv_test.F1_SCORE = tester.metrics.f1_score()
        csv_test.ACCURACY = tester.metrics.accuracy()
        csv_test.MIOU = tester.metrics.mIou()
        csv_test.TP = tester.metrics.tp
        csv_test.FP = tester.metrics.fp
        csv_test.TN = tester.metrics.tn
        csv_test.FN = tester.metrics.fn
        csv_test.append()
        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    features_list = ["nxnynz"]
    
    for features in features_list:
        cfg_path = f"/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept/exp/retTruss/{features}/config.py"
        test(features, cfg_path)