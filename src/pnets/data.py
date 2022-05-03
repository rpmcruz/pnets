from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import os

class Sydney(Dataset):
    def __init__(self, root, download, fold, transform=None):
        assert fold in ('train', 'test'), f'fold must be train or test, not {fold}'
        if download:
            url = 'http://www.acfr.usyd.edu.au/papers/data/sydney-urban-objects-dataset.tar.gz'
            download_and_extract_archive(url, root)
        folds = [0, 1, 2] if fold == 'train' else ['3']
        self.root = os.path.join(root, 'sydney-urban-objects-dataset')
        self.files = []
        for i in folds:
            with open(os.path.join(self.root, 'folds', f'fold{i}.txt')) as f:
                self.files += [line.rstrip() for line in f.readlines()]
        self.labels = sorted(list(set(f.split('.')[0] for f in self.files)))
        self.to_class = {label: k for k, label in enumerate(self.labels)}
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = os.path.join(self.root, 'objects', self.files[i])
        names = ['t', 'intensity', 'id', 'x', 'y', 'z', 'azimuth', 'range', 'pid']
        formats = ['int64', 'uint8', 'uint8', 'float32', 'float32', 'float32', 'float32', 'float32', 'int32']
        binType = np.dtype({'names': names, 'formats': formats})
        data = np.fromfile(filename, binType)
        P = np.vstack([data['x'], data['y'], -data['z']])
        if self.transform:
            P = self.transform(P)
        Y = self.to_class[self.files[i].split('.')[0]]
        return P, Y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('i', type=int)
    args = parser.parse_args()

    import plots
    ds = globals()[args.dataset]('data', False, 'train')
    P, Y = ds[args.i]
    print(ds.classes[Y])
    plot.plot3d(P)
