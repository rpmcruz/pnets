import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_true')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--feature-transform', action='store_true')
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
import pnets as pn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using:', device)

tr = pn.datasets.Sydney('data', args.download, 'train')
K = len(tr.classes)
tr = pn.datasets.Normalize(tr)
tr = pn.datasets.ResamplePoints(tr, args.npoints)
tr = DataLoader(tr, 32, True, num_workers=2)

model = pn.pointnet.PointNetCls(K).to(device)
summary(model)

sim_data = torch.rand(32, 3, 2500).to(device)
out, _, _ = model(sim_data)
print('output shape:', out.shape)

opt = torch.optim.Adam(model.parameters(), 1e-3)
ce_loss = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1} / {args.epochs}')
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for P, Y in tr:
        P = P.to(device)
        Y = Y.to(device)
        opt.zero_grad()

        Y_pred, trans, trans_feat = model(P)
        loss = F.nll_loss(Y_pred, Y)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        K_pred = torch.nn.functional.softmax(Y_pred, 1).argmax(1)
        #print(Y, K_pred)
        avg_acc += float((Y == K_pred).float().mean()) / len(tr)
    toc = time()
    print(f'- {toc-tic:.1f}s - Loss: {avg_loss} - Acc: {avg_acc}')
