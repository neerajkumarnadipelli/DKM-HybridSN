import torch
import torch.nn as nn
import torch.nn.functional as F
from .dkm_layers import DKMConv3d, DKMConv2d, DKMLinear
class OilSpillDKMNet(nn.Module):
    def _init_(self, output_units=2):
        super(OilSpillDKMNet, self)._init_()
        self.conv1 = DKMConv3d(1, 4, kernel_size=3, num_clusters=2, padding=1)
        self.conv2 = DKMConv3d(4, 8, kernel_size=3, num_clusters=4, padding=1)
        self.conv3 = DKMConv3d(8, 16, kernel_size=3, num_clusters=8, padding=1)
        self.conv4 = DKMConv2d(16, 32, kernel_size=3, num_clusters=16, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = None
        self.output_layer = None
        self.output_units = output_units
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        b, c, d, h, w = x.shape
        x = x.view(b, c, d * h, w)
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1)
        if self.fc1 is None:
            self.fc1 = DKMLinear(x.shape[1], 16, num_clusters=8).to(x.device)
        if self.output_layer is None:
            self.output_layer = DKMLinear(16, self.output_units, num_clusters=4).to(x.device)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(x)
        return self.output_layer(x)
