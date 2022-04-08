import matplotlib.pyplot as plt
import numpy as np

fname = "../output/out_10"
_x_size = 10
_y_size = 10
_z_size = 10

# https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return '#%02x%02x%02x' % (r, g, b)
    #return [int(r), int(g), int(b), 0.5]

def rgb2(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    #return '#%02x%02x%02x' % (r, g, b)
    return r/255, g/255, b/255, (b + r)/(2*255)

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
v_in = np.genfromtxt(fname)
#v = (np.arange(_x_size * _y_size * _z_size)) / (_x_size * _y_size * _z_size)

v = np.array([(rgb(np.min(v_in), np.max(v_in), i)) for i in v_in])
V=v.reshape(_x_size, _y_size, _z_size)

v2 = np.array([rgb2(np.min(v_in), np.max(v_in), i) for i in v_in])
V2=v2.reshape(V.shape + (4,))
size = np.concatenate(((np.array(V.shape)*2) - 1, (np.array([4]))))
colors2_2 = np.zeros(size, dtype=np.float32)
colors2_2[::2, ::2, ::2] = V2

voxels2 = explode(np.ones(V.shape))
colors2 = explode(V)

print(colors2_2)

(x, y, z) = shrink_gaps(voxels2)

voxels2[-1] = True

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, voxels2, facecolors=colors2_2, edgecolor='none')
#ax.voxels(x, y, z, voxels2, edgecolor='none', alpha=0.05)
#ax.voxels(voxels2, facecolors=colors2_2, edgecolor='k')

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
