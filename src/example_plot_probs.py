'''
Example of visually evaluating a semantic segmentation model.
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('fold', choices=['train', 'val', 'test'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--npoints', default=2500, type=int)
args = parser.parse_args()

from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import pnets as pn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

model = torch.load(args.model, map_location=device)

transform = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
)

ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, args.fold, transform)
ts = DataLoader(ts, 32, True, num_workers=4, pin_memory=True)

model.eval()
for P, Y in ts:
    with torch.no_grad():
        Ypred, _, _ = model(P.to(device))
        Ypred = F.softmax(Ypred, 1)

    for p, y, ypred in zip(P, Y, Ypred):
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(p[0], p[1], p[2], c=y, marker='.', alpha=0.2, edgecolors='none')
        ax.set_title('Ground-truth')

        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(p[0], p[1], p[2], c=ypred.argmax(0), marker='.', alpha=0.2, edgecolors='none')
        ax.set_title('Prediction')

        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(p[0], p[1], p[2], c=ypred.amax(0), marker='.', alpha=0.2, edgecolors='none')
        ax.set_title('Confidence')

        plt.show()
