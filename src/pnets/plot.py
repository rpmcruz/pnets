'''
Plotting utilities.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d(pts, c='k'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[0], pts[1], pts[2], c=c, marker='.', alpha=0.2, edgecolors='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plot_topview(pts, c='k'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pts[0], pts[1], c=c, marker='.', alpha=0.2, edgecolors='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax

def zoomin(ax, pts, q=2.5):
    xbounds = (np.percentile(pts[0], q), np.percentile(pts[0], 100-q))
    ybounds = (np.percentile(pts[1], q), np.percentile(pts[1], 100-q))
    zbounds = (np.percentile(pts[2], q), np.percentile(pts[2], 100-q))
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    if type(ax) == Axes3D:
        ax.set_zlim(*zbounds)

def show():
    plt.show()
