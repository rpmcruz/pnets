import torch

def IoU(ypred, y, K):
    assert len(ypred.shape) == 1, 'ypred must be a vector'
    assert len(y.shape) == 1, 'y must be a vector'
    ypred = ypred[None]
    y = y[None]
    kk = torch.arange(K, device=y.device)[:, None]
    cmii = torch.logical_and(ypred == kk, y == kk).sum(1)
    cmij = torch.logical_and(ypred == kk, y != kk).sum(1)
    cmki = torch.logical_and(ypred != kk, y == kk).sum(1)
    return cmii / (cmii + cmij + cmki)

def mIoU(ypred, y, K):
    return IoU(ypred, y, K).mean()
