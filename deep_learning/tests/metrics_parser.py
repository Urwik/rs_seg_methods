import os
import matplotlib.pyplot as plt
from datetime import datetime


class MetricsTxtParser():
    def __init__(self, file_path=''):
        self.file_path = file_path
        self.metrics_dict = {}
        self.best_metric = 0
        self.best_epoch = 0
        self.target_metric = 'Precision'

    def get_metrics(self, file_path):
        metrics = []
        current_metrics = {}

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('EPOCH:'):
                    if current_metrics:  # If there are metrics already parsed, save them before starting a new epoch
                        metrics.append(current_metrics)
                        current_metrics = {}
                    # current_metrics['ID'] = os.path.basename(os.path.dirname(file_path))
                    current_metrics['Epoch'] = int(line.split(':')[1].strip())
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'mIoU']:
                        current_metrics[key] = float(value)
                    elif key == 'Duration':
                        current_metrics[key] = value  # Keep duration as string, or convert to timedelta if needed

            if current_metrics:  # Don't forget to add the last epoch's metrics
                metrics.append(current_metrics)

        self.metrics_dict = metrics
        return self.metrics_dict

    def num_epocs(self):
        self.get_metrics(self.file_path)
        return len(self.metrics_dict)

class MetricsParser():
    
    def __init__(self):
        self.root_dir = ""
        self.exp_metrics_files = []
        self.metrics_dict = {}
        self.best_metric = 0
        self.best_epoch = 0
        self.experiment_path = ''
        self.best_experiment_path = ''
        self.target_metric = 'Precision'
        
    def set_root_dir(self, root_dir):
        self.root_dir = root_dir
        
    def set_target_metric(self, metric_name):
        self.target_metric = metric_name
    
    def PointNetBinSeg(self):
        self.root_dir = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNetBinSeg'
    
    def PointNet2BinSeg(self):
        self.root_dir = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNet2BinSeg'
        
    def MinkUNet34C(self):
        self.root_dir = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/MinkUNet34C'
        
    def PointTransformerV3(self):
        self.root_dir = '/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNet2BinSeg'

    def set_experiment(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.metrics_file_path = os.path.join(experiment_dir, 'metrics.txt')
        self.get_metrics(self.metrics_file_path)

    def best_experiment(self):

        self.exp_metrics_files = self.search_all_subdirs(self.root_dir)
        
        current_best_metric = 0
        
        for current_metrics_file in self.exp_metrics_files:
            self.get_metrics(current_metrics_file)
            self.get_best_metric_value()
            
            if self.best_metric > current_best_metric:
                current_best_metric = self.best_metric
                self.best_experiment_path = os.path.dirname(current_metrics_file)

        return self.best_experiment_path 


    def get_best_metric_value(self):
        for metric in self.metrics_dict:
            if metric[self.target_metric] > self.best_metric:
                self.best_metric = metric[self.target_metric]
                self.best_epoch = metric['Epoch']
        return self.best_metric, self.best_epoch


    def get_metrics(self, file_path):
        metrics = []
        current_metrics = {}

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('EPOCH:'):
                    if current_metrics:  # If there are metrics already parsed, save them before starting a new epoch
                        metrics.append(current_metrics)
                        current_metrics = {}
                    # current_metrics['ID'] = os.path.basename(os.path.dirname(file_path))
                    current_metrics['Epoch'] = int(line.split(':')[1].strip())
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'mIoU']:
                        current_metrics[key] = float(value)
                    elif key == 'Duration':
                        current_metrics[key] = value  # Keep duration as string, or convert to timedelta if needed

            if current_metrics:  # Don't forget to add the last epoch's metrics
                metrics.append(current_metrics)

        self.metrics_dict = metrics
        return self.metrics_dict


    def best_exp_metrics(self):
        self.get_metrics(self.best_experiment_path)
        

    def search_all_subdirs(root_dir):
        abs_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.basename(file) == 'metrics.txt':
                    file_path = os.path.join(root, file)
                    abs_paths.append(file_path)

        return abs_paths


    def plot_epochs_vs_metric(self, metric_name=''):
        if metric_name == '':
            metric_name = self.target_metric
            
        epochs = [metric['Epoch'] for metric in self.metrics_dict]
        metric_values = [metric[metric_name] for metric in self.metrics_dict]
        
        plt.plot(epochs, metric_values)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        plt.title(f'{metric_name} vs Epoch')
        plt.show()

    def plot_duration_vs_metric(self, metric_name=''):
        if metric_name == '':
            metric_name = self.target_metric
            
        durations = [metric['Duration'] for metric in self.metrics_dict]
        metric_values = [metric[metric_name] for metric in self.metrics_dict]
        
        plt.plot(durations, metric_values)
        plt.xlabel('Duration')
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        plt.title(f'{metric_name} vs Duration')
        plt.show()
        
    def get_mean_epoch_duration(self):
        durations = []
        for metric in self.metrics_dict:
            if 'Duration' not in metric:
                raise ValueError('Duration not found in metrics')
            else:
                time_obj = datetime.strptime(metric['Duration'], '%H:%M:%S.%f')
                total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
                hours = total_seconds / 3600
                durations.append(hours)
        mean_duration = sum(durations) / len(durations)
        return mean_duration

if __name__ == '__main__':

    expParser = MetricsParser()
    
    expParser.PointNetBinSeg()
    expParser.set_target_metric('Precision')
    expParser.set_experiment('/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/tests/PointNetBinSeg/240619181313/metrics.txt')
    expParser.get_metrics(expParser.experiment_path)
    # expParser.best_experiment()
    # expParser.best_exp_metrics()
    expParser.plot_epochs_vs_metric()
        