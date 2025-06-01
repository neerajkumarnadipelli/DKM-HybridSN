import torch
import torch.nn as nn
import torch.nn.functional as F
#DKM 3D Convolution
class DKMConv3d(nn.Module):
    def _init_(self, in_channels, out_channels, kernel_size, num_clusters, stride=1, padding=0, tau=0.05):
        super(DKMConv3d, self)._init_()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.num_clusters = num_clusters
        self.clusters = nn.Parameter(torch.randn(num_clusters, in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]))
        self.assign_logits = nn.Parameter(torch.randn(out_channels, num_clusters))
        self.bn = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x):

        A = F.softmax(self.assign_logits / self.tau, dim=1)

        W = A @ self.clusters

        W = W.view(self.assign_logits.shape[0], -1, *self.kernel_size)

        x = F.conv3d(x, W, stride=self.stride, padding=self.padding)

        return self.bn(x)

#DKM 2D Convolution
class DKMConv2d(nn.Module):
    def _init_(self, in_channels, out_channels, kernel_size, num_clusters, stride=1, padding=0, tau=0.05):
        super(DKMConv2d, self)._init_()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.num_clusters = num_clusters
        self.clusters = nn.Parameter(torch.randn(num_clusters, in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.assign_logits = nn.Parameter(torch.randn(out_channels, num_clusters))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        A = F.softmax(self.assign_logits / self.tau, dim=1)
        W = A @ self.clusters
        W = W.view(self.assign_logits.shape[0], -1, *self.kernel_size)
        x = F.conv2d(x, W, stride=self.stride, padding=self.padding)
        return self.bn(x)

#DKM Linear Layer
class DKMLinear(nn.Module):
    def _init_(self, in_features, out_features, num_clusters, tau=0.05):
        super(DKMLinear, self)._init_()
        self.tau = tau
        self.num_clusters = num_clusters
        self.clusters = nn.Parameter(torch.randn(num_clusters, in_features))
        self.assign_logits = nn.Parameter(torch.randn(out_features, num_clusters))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        A = F.softmax(self.assign_logits / self.tau, dim=1)
        W = A @ self.clusters
        x = F.linear(x, W)
        return self.bn(x)

