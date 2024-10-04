from .lovaszbinary import LovaszLoss
import torch.nn as nn

class BCELovaszLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszLoss(ignore_index)

    def forward(self, logits, labels):
        return self.bce(logits, labels) + self.lovasz(logits, labels, self.ignore_index)