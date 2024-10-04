import numpy as np
import sklearn.metrics as metrics

class Metrics():
    def __init__(self, pred =None, label=None):
        if label is None or pred is None:
            self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0
        else:
            self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(label, pred).ravel()

    def is_nan(self):
        return np.isnan(self.precision()) or np.isnan(self.recall()) or np.isnan(self.f1_score()) or np.isnan(self.accuracy()) or np.isnan(self.mIou())

    def reset_metrics(self):
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def len(self):
        return self.tn + self.fp + self.fn + self.tp

    def set_metrics(self, pred, label):
        if np.ndim(pred) > 1:
            pred = np.array(pred).flatten()
        if np.ndim(label) > 1:
            label = np.array(label).flatten()

        cm = metrics.confusion_matrix(label, pred).ravel()
        if len(cm) == 1:
            self.tn, self.fp, self.fn, self.tp = 0, 0, 0, cm[0]
        else:
            self.tn, self.fp, self.fn, self.tp = cm

    def precision(self):
        
        if self.tp == 0 or (self.tp == 0 and self.fp == 0):
            return 0
        else:
            return self.tp / (self.tp + self.fp)
    
    def recall(self):
        if self.tp == 0 or (self.tp == 0 and self.fn == 0):
            return 0
        else:
            return self.tp / (self.tp + self.fn)
        
    
    def f1_score(self):
        
        if self.precision() == 0 or self.recall() == 0 or (self.precision() == 0 and self.recall() == 0):
            return 0
        else:
            return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
    
    def accuracy(self):
        if self.tp == 0 and self.fp == 0 and self.fn == 0 and self.tn == 0:
            return 0
        else:
            return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)        

    
    def mIou(self):
        
        if self.tp == 0 or (self.tp == 0 and self.fp == 0 and self.fn == 0):
            return 0
        else:
            return self.tp / (self.tp + self.fp + self.fn)

    def get_as_string(self):
        string = f'Precision: {self.precision()} | Recall: {self.recall()} | F1 Score: {self.f1_score()} | Accuracy: {self.accuracy()} | mIoU: {self.mIou()}'
        return string

    def print(self, horizontal = False):
        if horizontal:
            print(f'Precision: {self.precision()} | Recall: {self.recall()} | F1 Score: {self.f1_score()} | Accuracy: {self.accuracy()} | mIoU: {self.mIou()}')
        else:
            print(f'\tPrecision: {self.precision()}')
            print(f'\tRecall:    {self.recall()}')
            print(f'\tF1 Score:  {self.f1_score()}')
            print(f'\tAccuracy:  {self.accuracy()}')
            print(f'\tmIoU:      {self.mIou()}')