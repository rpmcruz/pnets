'''
Datasets for classification and semantic segmentation. The points format is
channels first (i.e. 3xN), like PyTorch expects.
'''

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import os

class Sydney(Dataset):
    ''' Sydney Urban Object Dataset is a classification dataset. https://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml '''

    segmentation = False

    def __init__(self, root, fold, transform, download=False):
        assert fold in ('train', 'test'), f'fold must be train or test, not {fold}'
        if download:
            url = 'http://www.acfr.usyd.edu.au/papers/data/sydney-urban-objects-dataset.tar.gz'
            download_and_extract_archive(url, root)
        folds = [0, 1, 2] if fold == 'train' else ['3']
        self.root = os.path.join(root, 'sydney-urban-objects-dataset')
        self.transform = transform
        self.files = []
        for i in folds:
            with open(os.path.join(self.root, 'folds', f'fold{i}.txt')) as f:
                self.files += [line.rstrip() for line in f.readlines()]
        self.labels = sorted(list(set(f.split('.')[0] for f in self.files)))
        self.nclasses = len(self.labels)
        self.to_class = {label: k for k, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = os.path.join(self.root, 'objects', self.files[i])
        names = ['t', 'intensity', 'id', 'x', 'y', 'z', 'azimuth', 'range', 'pid']
        formats = ['int64', 'uint8', 'uint8', 'float32', 'float32', 'float32', 'float32', 'float32', 'int32']
        binType = np.dtype({'names': names, 'formats': formats})
        data = np.fromfile(filename, binType)
        P = np.stack((data['x'], data['y'], -data['z']))
        if self.transform:
            P, _ = self.transform(P, None)
        Y = self.to_class[self.files[i].split('.')[0]]
        return P, Y

class SemanticKITTI(Dataset):
    ''' Semantic KITTI is a dataset that includes semantic and instance segmentation, among others. Here we support semantic segmentation. http://www.semantic-kitti.org/ '''

    nclasses = 20  # or 19 if we ignore class 0
    classes_map = {  # like semantic-kitti-api, merge indistinguishable classes
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
        32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
        71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
        256: 5, 257: 5, 258: 4, 259: 5
    }
    segmentation = True

    def __init__(self, root, fold, transform):
        assert fold in ('train', 'test')
        self.root = os.path.join(root, 'semantic-kitti')
        self.transform = transform
        # as suggested by semantic-kitti, we use 00-10 (train), 11-21 (test)
        seqs = range(0, 10+1) if fold == 'train' else range(11, 21+1)
        self.files = []
        for seq in seqs:
            self.files += [(seq, f) for f in os.listdir(os.path.join(self.root, 'sequences', '%02d' % seq, 'velodyne'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        seq, fname = self.files[i]
        P_fname = os.path.join(self.root, 'sequences', '%02d' % seq, 'velodyne', fname)
        P = np.fromfile(P_fname, np.float32).reshape((-1, 4))[:, :3].T
        S_fname = os.path.join(self.root, 'sequences', '%02d' % seq, 'labels', fname[:-3] + 'label')
        S = np.fromfile(S_fname, np.uint16).reshape((-1, 2))[:, 0]
        S = np.array([self.classes_map[s] for s in S], np.int64)
        if self.transform:
            P, S = self.transform(P, S)
        return P, S

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('i', type=int)
    parser.add_argument('--topview', action='store_true')
    args = parser.parse_args()

    import plot
    ds = globals()[args.dataset]('/data', 'train')
    P, Y = ds[args.i]
    f = plot.plot_topview if args.topview else plot.plot3d
    c = Y if ds.segmentation else 'k'
    ax = f(P, c)
    plot.zoomin(ax, P)
    if not ds.segmentation:
        ax.set_title(f'class={Y}')
    plot.show()
