'''
Datasets for classification and semantic segmentation. The points format is
channels first (i.e. 3xN), like PyTorch expects.
'''

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import json
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

class EricyiShapeNet(Dataset):
    '''A ShapeNetCore version with segmentations.'''

    nclasses = 16
    labels = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']

    def __init__(self, root, fold, transform, segmentation, download=False):
        assert fold in ('train', 'val', 'test')
        if download:
            url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'
            download_and_extract_archive(url, root)
        self.root = os.path.join(root, 'ShapeNet', 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.transform = transform
        self.segmentation = segmentation
        catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        with open(catfile) as f:
            cat2id = {line.split()[1]: line.split()[0] for line in f}
        splitfile = os.path.join(self.root, 'train_test_split', f'shuffled_{fold}_file_list.json')
        filelist = json.load(open(splitfile))
        self.files = [f.split('/')[1:] for f in filelist]
        self.classes = [self.labels.index(cat2id[f[0]]) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        category, uuid = self.files[i]
        P_fname = os.path.join(self.root, category, 'points', uuid + '.pts')
        P = np.loadtxt(P_fname).astype(np.float32).T
        if self.segmentation:
            S_fname = os.path.join(self.root, category, 'points_label', uuid + '.seg')
            Y = np.loadtxt(S_fname).astype(np.int64)
        else:
            Y = self.classes[i]
        if self.transform:
            P, Y = self.transform(P, Y)
        return P, Y

class EricyiShapeNetClass(EricyiShapeNet):
    def __init__(self, root, fold, transform):
        super().__init__(root, fold, transform, False)

class EricyiShapeNetSeg(EricyiShapeNet):
    def __init__(self, root, fold, transform):
        super().__init__(root, fold, transform, True)

class Stanford3d(Dataset):
    map_classes = {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4, 'window': 5, 'door': 6, 'table': 7, 'chair': 8, 'sofa': 9, 'bookcase': 10, 'board': 11, 'clutter': 12}
    segmentation = True
    nclasses = 13

    def __init__(self, root, fold, transform):
        assert fold in ('train', 'test')
        self.root = os.path.join(root, 'Stanford3d', 'Stanford3dDataset_v1.2_Aligned_Version')
        self.transform = transform
        areas = [1, 2, 3, 4, 6] if fold == 'train' else [5]
        self.rooms = [(f'Area_{area}', room) for area in areas for room in os.listdir(os.path.join(self.root, f'Area_{area}'))]

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, i):
        area, room = self.rooms[i]
        segs = os.listdir(os.path.join(self.root, area, room, 'Annotations'))
        P = []
        S = []
        for seg in segs:
            pts = np.loadtxt(os.path.join(self.root, area, room, 'Annotations', seg), np.float32, usecols=[0, 1, 2])
            P.append(pts)
            S += [self.map_classes[seg.split('_')[0]]] * len(pts)
        P = np.concatenate(P).T
        S = np.array(S, np.int64)
        if self.transform:
            P, S = self.transform(P, S)
        return P, S

class Cache(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
        self.cache = [None] * len(self.ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.ds[i]
        P, S = self.cache[i]
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
