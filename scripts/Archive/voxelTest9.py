import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fname = "../output/out_10"
_x_size = 10//1
_y_size = 10//1
_z_size = 10//1

def rgb(minimum, maximum, value):
    norm = mpl.colors.Normalize(vmin=np.min(minimum), vmax=np.max(maximum))
    cmap = cm.bwr
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    (r, g, b, a) = m.to_rgba(value)
    return r, g, b, (1-g)/4

#https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html
def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html
def shrink_gaps(data):
    x, y, z = np.indices(np.array(data.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.005
    y[:, 0::2, :] += 0.005
    z[:, :, 0::2] += 0.005
    x[1::2, :, :] += 0.905
    y[:, 1::2, :] += 0.905
    z[:, :, 1::2] += 0.905
    return (x, y, z)

# STARAT HERE
v = np.genfromtxt(fname)
#v = v[0:(len(v)//2)]

data_shape = (_x_size, _y_size, _z_size)

v2 = np.array([rgb(np.min(v), np.max(v), i) for i in v])
V2=v2.reshape(data_shape + (4,))
size = np.concatenate(((np.array(data_shape)*2) - 1, (np.array([4]))))
colors2 = np.zeros(size, dtype=np.float32)
colors2[::2, ::2, ::2] = V2

voxels2 = explode(np.ones(data_shape))

(x, y, z) = shrink_gaps(voxels2)

voxels2[-1] = False

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, voxels2, facecolors=colors2, edgecolor='none')

plt.show()



















    
### prepare some coordinates
##x, y, z = np.indices((8, 8, 8))# from  ww  w  .  d em o  2 s  .  c  om
##
### draw cuboids in the top left and bottom right corners, and a link between
### them
##cube1 = (x < 3) & (y < 3) & (z < 3)
##cube2 = (x >= 5) & (y >= 5) & (z >= 5)
##link = abs(x - y) + abs(y - z) + abs(z - x) <= 2
##
### combine the objects into a single boolean array
##voxels = cube1 | cube2 | link
##
### set the colors of each object
##colors = np.empty(voxels.shape, dtype=object)
##colors[link] = 'red'
##colors[cube1] = 'blue'
##colors[cube2] = 'green'
##
##voxels2 = explode(voxels)
##colors2 = explode(colors)
##
##(x, y, z) = shrink_gaps(voxels2)
##
### and plot everything
##ax = plt.figure().add_subplot(projection='3d')
##ax.voxels(x, y, z, voxels2, facecolors=colors2, edgecolor='k')
##
##plt.show()
