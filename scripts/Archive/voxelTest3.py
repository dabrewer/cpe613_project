import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA
from matplotlib.cbook import _backports
from collections import defaultdict
import types

def voxels(self, *args, **kwargs):

    if len(args) >= 3:
        # underscores indicate position only
        def voxels(__x, __y, __z, filled, **kwargs):
            return (__x, __y, __z), filled, kwargs
    else:
        def voxels(filled, **kwargs):
            return None, filled, kwargs

    xyz, filled, kwargs = voxels(*args, **kwargs)

    # check dimensions
    if filled.ndim != 3:
        raise ValueError("Argument filled must be 3-dimensional")
    size = np.array(filled.shape, dtype=np.intp)

    # check xyz coordinates, which are one larger than the filled shape
    coord_shape = tuple(size + 1)
    if xyz is None:
        x, y, z = np.indices(coord_shape)
    else:
        x, y, z = (_backports.broadcast_to(c, coord_shape) for c in xyz)

    def _broadcast_color_arg(color, name):
        if np.ndim(color) in (0, 1):
            # single color, like "red" or [1, 0, 0]
            return _backports.broadcast_to(
                color, filled.shape + np.shape(color))
        elif np.ndim(color) in (3, 4):
            # 3D array of strings, or 4D array with last axis rgb
            if np.shape(color)[:3] != filled.shape:
                raise ValueError(
                    "When multidimensional, {} must match the shape of "
                    "filled".format(name))
            return color
        else:
            raise ValueError("Invalid {} argument".format(name))

    # intercept the facecolors, handling defaults and broacasting
    facecolors = kwargs.pop('facecolors', None)
    if facecolors is None:
        facecolors = self._get_patches_for_fill.get_next_color()
    facecolors = _broadcast_color_arg(facecolors, 'facecolors')

    # broadcast but no default on edgecolors
    edgecolors = kwargs.pop('edgecolors', None)
    edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

    # include possibly occluded internal faces or not
    internal_faces = kwargs.pop('internal_faces', False)

    # always scale to the full array, even if the data is only in the center
    self.auto_scale_xyz(x, y, z)

    # points lying on corners of a square
    square = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ], dtype=np.intp)

    voxel_faces = defaultdict(list)

    def permutation_matrices(n):
        """ Generator of cyclic permutation matices """
        mat = np.eye(n, dtype=np.intp)
        for i in range(n):
            yield mat
            mat = np.roll(mat, 1, axis=0)

    for permute in permutation_matrices(3):
        pc, qc, rc = permute.T.dot(size)
        pinds = np.arange(pc)
        qinds = np.arange(qc)
        rinds = np.arange(rc)

        square_rot = square.dot(permute.T)

        for p in pinds:
            for q in qinds:
                p0 = permute.dot([p, q, 0])
                i0 = tuple(p0)
                if filled[i0]:
                    voxel_faces[i0].append(p0 + square_rot)

                # draw middle faces
                for r1, r2 in zip(rinds[:-1], rinds[1:]):
                    p1 = permute.dot([p, q, r1])
                    p2 = permute.dot([p, q, r2])
                    i1 = tuple(p1)
                    i2 = tuple(p2)
                    if filled[i1] and (internal_faces or not filled[i2]):
                        voxel_faces[i1].append(p2 + square_rot)
                    elif (internal_faces or not filled[i1]) and filled[i2]:
                        voxel_faces[i2].append(p2 + square_rot)

                # draw upper faces
                pk = permute.dot([p, q, rc-1])
                pk2 = permute.dot([p, q, rc])
                ik = tuple(pk)
                if filled[ik]:
                    voxel_faces[ik].append(pk2 + square_rot)

    # iterate over the faces, and generate a Poly3DCollection for each voxel
    polygons = {}
    for coord, faces_inds in voxel_faces.items():
        # convert indices into 3D positions
        if xyz is None:
            faces = faces_inds
        else:
            faces = []
            for face_inds in faces_inds:
                ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                face = np.empty(face_inds.shape)
                face[:, 0] = x[ind]
                face[:, 1] = y[ind]
                face[:, 2] = z[ind]
                faces.append(face)

        poly = art3d.Poly3DCollection(faces,
            facecolors=facecolors[coord],
            edgecolors=edgecolors[coord],
            **kwargs
        )
        self.add_collection3d(poly)
        polygons[coord] = poly

    return polygons


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
    return (r, g, b)

#https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html
def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html
def shrink_gaps(data):
    x, y, z = np.indices(np.array(data.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95
    return (x, y, z)

# STARAT HERE
#v = np.genfromtxt(fname)
v = (np.arange(_x_size * _y_size * _z_size)) / (_x_size * _y_size * _z_size)
print(len(v))
print(v)

v = np.array([str(i) for i in v])
#v = np.array([str((i - np.min(v)) / (np.max(v) - np.min(v))) for i in v])
##print(v)

V=v.reshape(_x_size, _y_size, _z_size)
print(V)

colors = np.empty(V.shape + (4,), dtype=np.float32)
alpha = .5
colors[0] = [1, 0, 0, alpha]
colors[1] = [0, 1, 0, alpha]
colors[2] = [0, 0, 1, alpha]
colors[3] = [1, 1, 0, alpha]
colors[4] = [0, 1, 1, alpha]
colors[5] = [1, 0, 0, alpha]
colors[6] = [0, 1, 0, alpha]
colors[7] = [0, 0, 1, alpha]
colors[8] = [1, 1, 0, alpha]
colors[9] = [0, 1, 1, alpha]

filled = np.ones(V.shape, dtype=bool)

#voxels2 = explode(np.ones(V.shape))
colors2 = explode(colors)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(filled, facecolors=colors, edgecolors='k')



#voxels2 = explode(np.ones(V.shape))
#colors2 = explode(V)

#(x, y, z) = shrink_gaps(voxels2)

# and plot everything
#ax = plt.figure().add_subplot(projection='3d')
#ax.voxels(x, y, z, voxels2, facecolors=colors2, edgecolor='k')

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
