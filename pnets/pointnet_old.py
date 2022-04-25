from torch import nn
from torch.nn import functional as F
import torch

'''
PointNet paper: https://arxiv.org/abs/1612.00593

One thing to keep in mind is that the paper says that "Batchnorm is used for all
layers". Yet, in some applications, the number of points differs for each object
which means that we must train with batch_size=1 (or perform padding), which is
not compatible with batch norm (std=0 for n=1). My solution was to not apply
batch norm whenever the number of samples is 1.
'''

def tnet_regularizer(m):
    I = torch.eye(m.shape[1], device=m.device)[None]
    return torch.mean(torch.norm(torch.bmm(m, m.transpose(2, 1)) - I, dim=(1, 2)))

class MLP(nn.Module):
    def __init__(self, input_channel, channels, shared):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for prev, next in zip([input_channel] + channels[:-1], channels):
            if shared:
                self.mlps.append(nn.Conv1d(prev, next, 1))
            else:
                self.mlps.append(nn.Linear(prev, next))
            self.bns.append(nn.BatchNorm1d(next))

    def forward(self, x):
        for mlp, bn in zip(self.mlps, self.bns):
            x = mlp(x)
            if x.shape[0] > 1:  # batch norm requires n>1 (see comment above)
                x = bn(x)
            x = F.relu(x)
        return x

def max_pool(x):
    return torch.max(x, 2).values

class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.mlp1 = MLP(k, [64, 128, 1024], True)
        self.mlp2 = MLP(1024, [512, 256, k*k], False)
        self.k = k

    def forward(self, x):
        m = self.mlp1(x)
        m = max_pool(m)
        m = self.mlp2(m)
        m = m.view(-1, self.k, self.k)
        x = torch.bmm(m, x)
        return m, x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(3)
        self.mlp1 = MLP(3, [64, 64], True)
        self.feature_transform = TNet(64)
        self.mlp2 = MLP(64, [64, 128, 1024], True)

    def forward(self, x):
        m1, x = self.input_transform(x)
        x = self.mlp1(x)
        m2, f = self.feature_transform(x)
        g = self.mlp2(f)
        g = max_pool(g)
        return [m1, m2], f, g

class Classifier(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.backbone = Backbone()
        self.mlp = MLP(1024, [512, 256, k], False)

    def forward(self, x):
        mm, _, g = self.backbone(x)
        x = self.mlp(g)
        return mm, x

class Segmentation(nn.Module):
    def __init__(self, m):
        self.backbone = Backbone()
        self.mlp = MLP(1088, [516, 256, 128, m], True)

    def forward(self, x):
        mm, f, g = self.backbone(x)
        g = g[:, :, None].repeat(1, 1, f.shape[2])
        x = torch.cat((f, g), 1)
        x = self.mlp(x)
        return mm, x
