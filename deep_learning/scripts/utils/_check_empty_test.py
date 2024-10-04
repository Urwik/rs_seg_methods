import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.metrics_parser import MetricsTxtParser
# from tests.metrics_parser import MetricsTxtParser
# from ...tests.metrics_parser import MetricsTxtParser


def get_num_of_epochs(model_dir: str):
    parser = MetricsTxtParser(os.path.join(model_dir, 'metrics.txt'))
    print(f"{os.path.basename(model_dir)}: {parser.num_epocs()}")
    return parser.num_epocs()

def check_empty_test(model_dir: str, rm_empty: bool = True):
    if os.path.exists(os.path.join(model_dir, 'model.pth')):
        
        if os.path.exists(os.path.join(model_dir, 'metrics.txt')):
            parser = MetricsTxtParser(os.path.join(model_dir, 'metrics.txt'))

            if parser.num_epocs() > 5:
                return True
            else:
                print(f"Less than 5 epochs: {os.path.basename(model_dir)}")
                if rm_empty: shutil.rmtree(model_dir)
                return False
        else:
            print(f"No metrics saved: {os.path.basename(model_dir)}")
            if rm_empty: shutil.rmtree(model_dir)
            return False
        
    else:
        print(f"No model saved: {os.path.basename(model_dir)}")
        if rm_empty: shutil.rmtree(model_dir)
        return False
    
if __name__ == '__main__':
    
    MODEL = 'PointNet2BinSeg'

    MODEL_EXPERIMENTS_DIR = f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/{MODEL}'    
    for model_dir in os.listdir(MODEL_EXPERIMENTS_DIR):
        tmp_model_dir = os.path.join(MODEL_EXPERIMENTS_DIR, model_dir)
        if os.path.isdir(tmp_model_dir):
            try:
                check_empty_test(tmp_model_dir, rm_empty=False)
                # get_num_of_epochs(os.path.join(MODEL_EXPERIMENTS_DIR, model_dir))
            
            except:
                print(f"Error while trying to read from: {tmp_model_dir}")