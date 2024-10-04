import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.metrics_parser import MetricsTxtParser
# from tests.metrics_parser import MetricsTxtParser
# from ...tests.metrics_parser import MetricsTxtParser
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_END = "\033[0m"

def check_threshold(model_dir: str):
    if os.path.exists(os.path.join(model_dir, 'threshold.npy')):
        return True
    else:
        print(f"No threshold saved: {os.path.basename(model_dir)}")
        return False
    
def check_model(model_dir: str):
    if os.path.exists(os.path.join(model_dir, 'model.pth')):
        return True
    else:
        print(f"No model saved: {os.path.basename(model_dir)}")
        return False
    
def check_metrics(model_dir: str):
    if os.path.exists(os.path.join(model_dir, 'metrics.txt')):
        parser = MetricsTxtParser(os.path.join(model_dir, 'metrics.txt'))
        if parser.num_epocs() < 5:
            print(f"{ANSI_YELLOW}Warning: Metrics has less than 5 epochs{ANSI_END}")
        return True
    else:
        print(f"No metrics saved: {os.path.basename(model_dir)}")
        return False
    
def check_config(model_dir: str):
    if os.path.exists(os.path.join(model_dir, 'config.yaml')):
        return True
    else:
        print(f"No config saved: {os.path.basename(model_dir)}")
        return False



def check_test(model_dir: str):
    if check_threshold(model_dir) and check_model(model_dir) and check_metrics(model_dir) and check_config(model_dir):
        return True
    else:
        return False
    
if __name__ == '__main__':
    
    MODEL = 'PointNet2BinSeg'

    MODEL_EXPERIMENTS_DIR = f'/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/{MODEL}'    
    for model_dir in os.listdir(MODEL_EXPERIMENTS_DIR):
        tmp_model_dir = os.path.join(MODEL_EXPERIMENTS_DIR, model_dir)
        if os.path.isdir(tmp_model_dir):
            try:
                if check_test(tmp_model_dir):
                    print(f"{model_dir}: {ANSI_GREEN}OK{ANSI_END}")
                else:
                    print(f"{model_dir}: {ANSI_RED}ERROR{ANSI_END}")
                    print(f"\tModel: {check_model(tmp_model_dir)}")
                    print(f"\tThreshold: {check_threshold(tmp_model_dir)}")
                    print(f"\tMetrics: {check_metrics(tmp_model_dir)}")
                    print(f"\tConfig: {check_config(tmp_model_dir)}")

            except:
                print(f"Error while trying to read from: {tmp_model_dir}")