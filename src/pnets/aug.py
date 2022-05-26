'''
Point-cloud augmentation pipeline routines.

Each augmentation routine `f(P, S)` must have as input the points and segmentation, and output the same. The segmentation can be `None` if there are no segmentations.
'''

import numpy as np

def rotation_matrix(axis, angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    if axis == 'X':
        R = np.array((
            (1, 0, 0),
            (0, cos_angle, -sin_angle),
            (0, sin_angle, cos_angle)
        ), np.float32)
    elif axis == 'Y':
        R = np.array((
            (cos_angle, 0, sin_angle),
            (0, 1, 0),
            (-sin_angle, 0, cos_angle)
        ), np.float32)
    else:  # axis == 'Z':
        R = np.array((
            (cos_angle, -sin_angle, 0),
            (sin_angle, cos_angle, 0),
            (0, 0, 1)
        ), np.float32)
    return R

def Compose(*l):
    ''' Receives a sequence of augmentation routines. '''
    def f(P, S):
        for t in l:
            P, S = t(P, S)
        return P, S
    return f

def Normalize():
    ''' Normalization to [0,1]. Formula: `(P-center)/max_value`. '''
    def f(P, S):
        center = np.mean(P, 0, keepdims=True)
        max_value = np.max(np.abs(P-center), 0, keepdims=True)
        P = (P-center)/max_value
        return P, S
    return f

def Resample(npoints):
    ''' Undersamples/oversamples in order to keep the number of points fixed to the specified number `npoints`. This is useful to keep each batch the same size. '''
    def f(P, S):
        ix = np.random.choice(P.shape[1], npoints)
        P = P[:, ix]
        if S is None:
            return P, S
        S = S[ix]
        return P, S
    return f

def Jitter(sdev=0.02):
    ''' Adds Gaussian noise with the given `sdev` deviation. '''
    def f(P, S):
        noise = np.random.randn(*P.shape)*sdev
        P += noise
        return P, S
    return f

def RandomRotation(axis, angle_min, angle_max):
    ''' Performs a rotation along the given `axis` (X, Y, Z), randomly in the range `[angle_min, angle_max]` in radians. '''
    assert axis in ('X', 'Y', 'Z')
    def f(P, S):
        angle = np.random.rand()*(angle_max-angle_min) + angle_min
        R = rotation_matrix(axis, angle)
        P = np.dot(R, P)
        return P, S
    return f

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    args = parser.parse_args()
    import data, plot
    t = RandomRotation('Z', 0, 2*np.pi)
    ds = data.Sydney(args.datadir, 'train', t)
    P, Y = ds[10]
    print(ds.labels[Y])
    plot.plot3d(P)
