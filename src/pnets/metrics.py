import torch

def accuracy(ypred, y):
    return (ypred == y).mean()

def IoU(ypred, y, K):
    assert len(ypred.shape) == 1, 'ypred must be a vector'
    assert len(y.shape) == 1, 'y must be a vector'
    assert torch.all(torch.logical_and(ypred >= 0, ypred < K)), 'ypred must be probabilities (did you forget the softmax?)'
    ypred = ypred[None]
    y = y[None]
    kk = torch.arange(K, device=y.device)[:, None]
    cmii = torch.logical_and(ypred == kk, y == kk).sum(1)
    cmij = torch.logical_and(ypred == kk, y != kk).sum(1)
    cmki = torch.logical_and(ypred != kk, y == kk).sum(1)
    return cmii / (cmii + cmij + cmki)

def mIoU(ypred, y, K):
    return IoU(ypred, y, K).mean()
