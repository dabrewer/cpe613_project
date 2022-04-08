import numpy as np
import matplotlib.pyplot as plt

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

spatial_axes = [5, 5, 5]
filled = np.ones(spatial_axes, dtype=bool)

colors = np.empty(spatial_axes + [4], dtype=np.float32)
alpha = .5
colors[0] = [1, 0, 0, alpha]
colors[1] = [0, 1, 0, alpha]
colors[2] = [0, 0, 1, alpha]
colors[3] = [1, 1, 0, alpha]
colors[4] = [0, 1, 1, alpha]

print(np.array([4]))
size = np.concatenate(((np.array(filled.shape)*2) - 1, (np.array([4]))))
colors2_2 = np.zeros(size, dtype=colors.dtype)
colors2_2[::2, ::2, ::2] = colors

print(colors2_2)
print(size)

filled = explode(filled)

fig = plt.figure()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
filled[-1] = False
ax.voxels(filled, facecolors=colors2_2, edgecolors='k')

plt.show()
