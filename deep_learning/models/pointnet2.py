import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class PointNet2BinSeg(nn.Module):
    def __init__(self, n_feat, device='cpu', dropout_=True):
        super(PointNet2BinSeg, self).__init__()
        self.device = device
        self.dropout = dropout_
        out_classes = 1
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, n_feat + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_classes, 1)

    def get_name(self):
        return 'PointNet2BinSeg'

    def forward(self, x):
        batchsize = x[0].size()[0]
        n_pts = x[0].size()[1]

        coords = x[0].to(self.device, dtype=torch.float32)
        features = x[1].to(self.device, dtype=torch.float32)
        # DATA SHOULD BE BATCH X N_FEAT/N_COORDS X N_PTS

        l0_points = features.transpose(2, 1)
        l0_xyz = coords.transpose(2, 1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Enable dropout for training and disable for test
        if self.dropout:
            x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        else:
            x = F.relu(self.bn1(self.conv1(l0_points)))

        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        x = x.view(batchsize, n_pts)
        return x #, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = PointNet2BinSeg(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))