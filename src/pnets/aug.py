# Augmentation techniques

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
    def f(P, S=None):
        if S is None:
            for t in l:
                P = t(P)
            return P
        for t in l:
            P, S = t(P, S)
        return P, S
    return f

def Normalize():
    def f(P, S=None):
        center = np.mean(P, 0, keepdims=True)
        max_value = np.max(np.abs(P-center), 0, keepdims=True)
        P = (P-center)/max_value
        if S is None:
            return P
        S = (S-center)/max_value
        return P, S
    return f

def Resample(npoints):
    def f(P, S=None):
        ix = np.random.choice(P.shape[1], npoints)
        P = P[:, ix]
        if S is None:
            return P
        S = S[:, ix]
        return P, S
    return f

def Jitter(sdev=0.02):
    def f(P, S=None):
        noise = np.random.randn(*P.shape)*sdev
        P += noise
        if S is None:
            return P
        S += noise
        return P, S
    return f

def RandomRotation(axis, angle_min, angle_max):
    def f(P, S=None):
        angle = np.random.rand()*(angle_max-angle_min) + angle_min
        R = rotation_matrix(axis, angle)
        P = np.dot(R, P)
        if S is None:
            return P
        S = np.dot(R, S)
        return P, S
    return f

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    import data, plot
    t = RandomRotation('Z', 0, 2*np.pi)
    ds = data.Sydney(args.datadir, args.download, 'train', t)
    P, Y = ds[10]
    print(ds.labels[Y])
    plot.plot3d(P)
