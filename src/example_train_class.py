'''
Example of training a classifier (uses the Sydney dataset).
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='/data')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--feature-transform', action='store_true')
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
import numpy as np
import pnets as pn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation
aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
    pn.aug.Jitter(),
    pn.aug.RandomRotation('Z', 0, 2*np.pi),
)
tr = pn.data.Sydney(args.datadir, 'train', aug)
K = tr.nclasses
#tr = torch.utils.data.Subset(tr, range(10))  # DEBUG
tr = DataLoader(tr, 32, True, num_workers=4)

# create the model
model = pn.pointnet.PointNetCls(K).to(device)
summary(model)
print('model output:', model(torch.ones((10, 3, 2500), device=device))[0].shape)

opt = torch.optim.Adam(model.parameters(), 1e-3)
ce_loss = torch.nn.CrossEntropyLoss()

# train the model
model.train()
for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1} / {args.epochs}')
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for P, Y in tr:
        P = P.to(device)
        Y = Y.to(device)

        Y_pred, trans, trans_feat = model(P)
        loss = ce_loss(Y_pred, Y)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        K_pred = F.softmax(Y_pred, 1).argmax(1)
        avg_acc += float((Y == K_pred).float().mean()) / len(tr)
    toc = time()
    print(f'- {toc-tic:.1f}s - Loss: {avg_loss} - Acc: {avg_acc}')

torch.save(model, 'model-sydney.pth')
