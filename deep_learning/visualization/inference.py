import sys
import warnings
import requests
import yaml
import os
import pandas as pd


sys.path.append('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation')

from engines.trainers.pnTrainer import pnTester 
from engines.trainers.pn2Trainer import pn2Tester 
from engines.trainers.mkTrainer import mkTester 



clouds_map = {
    "orto":     ["00147", "00360", "00524", "00227", "00345", "00181"],
    "crossed":  ["00147", "00360", "00524", "00227", "00345", "00181"],
    "00": ["00000", "00001", "00002", "00003", "00004", "00005"],
    "01": ["00000", "00001", "00002", "00003", "00004", "00005"],
    "02": ["00000", "00001", "00002", "00003", "00004"],
    "03": ["00000", "00001", "00002", "00003", "00004", "00005"]
}


def max_miou_model_and_dataset():
    df = pd.read_csv("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/experiments/test.csv")
    
    results = []

    # Group by MODEL and DATASET
    grouped = df.groupby(['MODEL', 'DATASET_DIR'])

    # Iterate over each group
    for (model, dataset), group in grouped:
        # Find the row with the highest MIOU
        max_miou_row = group.loc[group['MIOU'].idxmax()]
        # Extract the EXPERIMENT_ID
        experiment_id = max_miou_row['EXPERIMENT_ID']
        # Append the result to the list
        results.append({
            'MODEL': model,
            'DATASET': dataset,
            'EXPERIMENT_ID': experiment_id,
            'MIOU': max_miou_row['MIOU']
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    with open('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/config/test.yaml', 'r') as file:
        test_config = yaml.safe_load(file)

    TEST_DIR = os.path.abspath("/home/arvc/Fran/datasets/complex_structure/")
    OUT_DIR = os.path.abspath("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/visualization/results/")
    
    tests_dir = "/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/"
    
    DF = max_miou_model_and_dataset()
    
    for index,row in DF.iterrows():
        exp_id = row['EXPERIMENT_ID']
        model = row['MODEL']
        dataset = row['DATASET']  
        
        exp_id = str(exp_id)
        dataset = str(dataset)
        if dataset == "test":
            continue
        
        
        model_dir = os.path.join(tests_dir, model, exp_id)
        
        if model == "PointNetBinSeg":
            Tester = pnTester
            suffix = 'ply_xyzln_fixedSize'
        elif model == "PointNet2BinSeg":
            Tester = pn2Tester
            suffix = 'ply_xyzln_fixedSize'
        elif model == "MinkUNet34C":
            Tester = mkTester
            suffix = 'ply_xyzln'
            
        # if model == "MinkUNet34C":
        test_config["model_dir"] = model_dir
        test_config['device'] = 'cuda:1'
        test_config['save_infered_clouds'] = True
        test_config['dataset_dir'] = os.path.join(TEST_DIR, dataset, suffix)
        test_config['output_dir'] = os.path.join(OUT_DIR, model, dataset)
        test_config['enable_metrics'] = False
            
        current_dataset = []
        for cloud in clouds_map[dataset]:
            cloud_path = os.path.join(test_config['dataset_dir'], cloud)
            cloud_path = cloud_path + ".ply"
            current_dataset.append(cloud_path)
                

        print("#"*50)
        print(f"Testing model: {model} - Experiment ID: {exp_id} - Testing dataset: {dataset}")
        print("#"*50)        
        tester = Tester(test_config_node= test_config)
        tester.test_dataset.dataset = current_dataset
        tester.build_dataloader()
        
        tester.test()
        
    print("Sending notification...")
    header = f'Test Completed: {tester.model.__class__.__name__}'
    message = f'Precision: {tester.test_metrics.precision()}\nRecall: {tester.test_metrics.recall()}\nF1 Score: {tester.test_metrics.f1_score()}\nmIoU: {tester.test_metrics.mIou()}'
    requests.post("https://ntfy.sh/arvc_train_fran", data=message.encode(encoding='utf-8'), headers={ "Title": header })


    
    


    