import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[0], pts[1], pts[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
