import csv 
import os
import ast
import pandas as pd

train_config_header = [
    "EXPERIMENT_ID",
    "DEVICE",
    "MODEL",
    "SEQUENCES",
    "FEATURES",
    "NORMALIZATION",
    "OPTIMIZER",
    "SCHEDULER",
    "THRESHOLD_METHOD",
    "TERMINATION_CRITERIA",
    "PRECISION",
    "VOXEL_SIZE",
    "MIOU",
    "RECALL",
    "F1_SCORE",
    "ACCURACY",
    "TP",
    "FP",
    "TN",
    "FN",
    "BEST_EPOCH",
    "NUM_EPOCHS",
    "DURATION"    
    ]

class csvTrainStruct():
    def __init__(self):
        self.output_file = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/experiments/train.csv'
        self.EXPERIMENT_ID = None
        self.DEVICE = None
        self.MODEL = None
        self.SEQUENCES = None
        self.FEATURES = None
        self.NORMALIZATION = None
        self.OPTIMIZER = None
        self.SCHEDULER = None
        self.THRESHOLD_METHOD = None
        self.TERMINATION_CRITERIA = None
        self.PRECISION = None
        self.VOXEL_SIZE = None
        self.MIOU = None
        self.RECALL = None
        self.F1_SCORE = None
        self.ACCURACY = None
        self.TP = None
        self.FP = None
        self.TN = None
        self.FN = None
        self.BEST_EPOCH = None
        self.NUM_EPOCHS = None
        self.DURATION = None

    def set_output_file(self, output_file):
        self.output_file = output_file
        
    def to_list(self):
        return [
            self.EXPERIMENT_ID,
            str(self.DEVICE),
            str(self.MODEL),
            str(self.SEQUENCES),
            str(self.FEATURES),
            self.NORMALIZATION,
            str(self.OPTIMIZER),
            str(self.SCHEDULER),
            str(self.THRESHOLD_METHOD),
            str(self.TERMINATION_CRITERIA),
            self.PRECISION,
            self.VOXEL_SIZE,
            self.MIOU,
            self.RECALL,
            self.F1_SCORE,
            self.ACCURACY,
            self.TP,
            self.FP,
            self.TN,
            self.FN,
            self.BEST_EPOCH,
            self.NUM_EPOCHS,
            self.DURATION
        ]
            
    def append(self):
        data = self.to_list()

        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(train_config_header)

        with open(self.output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)




test_config_header = [
    "EXPERIMENT_ID",
    "MODEL",
    "FEATURES",
    "VOXEL_SIZE",
    "GRID_SIZE",
    "OPTIMIZER",
    "THRESHOLD_METHOD",
    "TERMINATION_CRITERIA",
    "SCHEDULER",
    "NORMALIZATION",
    "DEVICE",
    "SEQUENCES",
    "BEST_EPOCH",
    "EPOCH_DURATION",
    "INFERENCE_DURATION",
    "PRECISION",
    "RECALL",
    "F1_SCORE",
    "ACCURACY",
    "MIOU",
    "TP",
    "FP",
    "TN",
    "FN"
    ]

class csvTestStruct():
    def __init__(self):
        self.output_file = '/home/fran/workspaces/nn_ws/binary_segmentation/experiments/test.csv'
        self.EXPERIMENT_ID = 0
        self.DEVICE = 'None'
        self.MODEL = 'None'
        self.SEQUENCES = 'None'
        self.FEATURES = 'None'
        self.NORMALIZATION = 'None'
        self.OPTIMIZER = 'None'
        self.SCHEDULER = 'None'
        self.THRESHOLD_METHOD = 'None'
        self.TERMINATION_CRITERIA = 'None'
        self.VOXEL_SIZE = 0
        self.GRID_SIZE = 0
        self.BEST_EPOCH = 0
        self.EPOCH_DURATION = 0
        self.INFERENCE_DURATION = 0
        self.PRECISION = 0
        self.RECALL = 0
        self.F1_SCORE = 0
        self.ACCURACY = 0
        self.MIOU = 0
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.TEST_DATASET = 'None'
        self.df = pd.read_csv(self.output_file)
        

    def set_output_file(self, output_file):
        self.output_file = output_file
    
    
    def filter(self, label, value):
        print("Filtering by ", label)

        if label in self.df.columns:
            if type(value) == list:
                self.df = self.df[self.df[label].apply(lambda x: any(str(item) in x for item in value))]
            else:
                self.df = self.df[self.df[label] == value]

        else: 
            print(f"ELEMENT: {label} not found in the dataframe header")

    def to_list(self):
        return [
            int(self.EXPERIMENT_ID),
            str(self.MODEL),
            str(self.FEATURES),
            float(self.VOXEL_SIZE),
            float(self.GRID_SIZE),
            str(self.OPTIMIZER),
            str(self.THRESHOLD_METHOD),
            str(self.TERMINATION_CRITERIA),
            str(self.SCHEDULER),
            str(self.NORMALIZATION),
            str(self.DEVICE),
            str(self.SEQUENCES),
            int(self.BEST_EPOCH),
            float(self.EPOCH_DURATION),
            float(self.INFERENCE_DURATION),
            float(self.PRECISION),
            float(self.RECALL),
            float(self.F1_SCORE),
            float(self.ACCURACY),
            float(self.MIOU),
            int(self.TP),
            int(self.FP),
            int(self.TN),
            int(self.FN),
            str(self.TEST_DATASET)
        ]
            
    def append(self):
        data = self.to_list()

        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(test_config_header)

        with open(self.output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)


class csvExperimentQueue():
    
    def __init__(self, output_file):
        self.output_file = output_file
        self.header=[]
        self.data = []
    
    def createCSV(self):
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as file:
               self.writer = csv.writer(file)
        
    def appendData(self, data):
        with open(self.output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            
    def getFirstRow(self):
        with open(self.output_file) as file:
            reader = csv.reader(file)
            return next(reader)
            # for first_row in reader:
            #     return [float(first_row[0]), ast.literal_eval(first_row[1]), str(first_row[2]), str(first_row[3])]

    def getAsList(self):
        data_list = []
        with open(self.output_file) as file:
            reader = csv.reader(file)
            for row in reader:
                row_list = []
                row_list.append(float(row[0]))
                row_list.append(ast.literal_eval(row[1]))
                row_list.append(str(row[2]))
                row_list.append(str(row[3]))
                data_list.append(row_list)
        
        return data_list
    
    def get_num_rows_in_csv(self):
        with open(self.output_file, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
        return row_count
    
    def removeRowByIndex(self, index):
        # Step 1: Read all rows into a list
        data_list = []
        with open(self.output_file, 'r') as file:
            reader = csv.reader(file)
            data_list = list(reader)
        
        # Step 2: Remove the row at the specified index
        if 0 <= index < len(data_list):
            data_list.pop(index)
        else:
            print("Index out of range")
            return
        
        # Step 3: Write the modified list back to the CSV
        with open(self.output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(data_list)
