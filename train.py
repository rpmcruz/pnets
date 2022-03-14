import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_true')
parser.add_argument('--epochs', default=50, type=int)
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
from time import time
import pnets as pn

tr = pn.datasets.Sydney('data', args.download, 'train')
K = len(tr.classes)
tr = pn.datasets.Normalize(tr)
tr = DataLoader(tr, 1, True)

model = pn.pointnet.Classifier(K).cuda()
summary(model)
print('output shape:', model(torch.zeros((1, 3, 1000), device='cuda'))[1].shape)

opt = torch.optim.Adam(model.parameters(), 1e-4)
ce_loss = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1} / {args.epochs}')
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for P, Y in tr:
        P = P.cuda()
        Y = Y.cuda()
        opt.zero_grad()
        mm, Y_pred = model(P)
        loss = ce_loss(Y_pred, Y)
        loss += sum(pn.pointnet.tnet_regularizer(m) for m in mm)
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        K_pred = torch.nn.functional.softmax(Y_pred, 1).argmax(1)
        avg_acc += float((Y == K_pred).float().mean()) / len(tr)
    toc = time()
    print(f'- {toc-tic:.1f}s - Loss: {avg_loss} - Acc: {avg_acc}')
