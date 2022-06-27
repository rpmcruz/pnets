'''
Example of evaluating a segmentation model.
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('fold', choices=['train', 'val', 'test'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--npoints', default=2500, type=int)
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import numpy as np
import pnets as pn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation
aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
)
ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, args.fold, aug)
K = ts.nclasses
#ts = torch.utils.data.Subset(ts, range(10))  # DEBUG
ts = DataLoader(ts, 128, num_workers=4, pin_memory=True)

# load the model
model = torch.load(args.model, map_location=device)

# evaluate the model
model.eval()
KK = []
KK_pred = []
tic = time()
avg_acc = 0
for P, Y in tqdm(ts):
    P = P.to(device)
    Y = Y.to(device)
    with torch.no_grad():
        Y_pred, _, _ = model(P)
    K_pred = F.softmax(Y_pred, 1).argmax(1)
    avg_acc += float((Y == K_pred).float().mean()) / len(ts)
    KK.append(Y.view(-1))
    KK_pred.append(K_pred.view(-1))
toc = time()
print(f'Time: {toc-tic:.1f}s')
print(f'Acc: {avg_acc}')
KK = torch.cat(KK)
KK_pred = torch.cat(KK_pred)
print('Acc per class:', pn.metrics.accuracy_per_class(KK_pred, KK).cpu().numpy())
print('IoU:', pn.metrics.IoU(KK_pred, KK, K).cpu().numpy())
print('mIoU:', pn.metrics.mIoU(KK_pred, KK, K).cpu().numpy())
