import matplotlib.pyplot as plt
import numpy as np

fname = "../output/out_f_10"
_x_size = 20
_y_size = 20
_z_size = 20

f_in = np.genfromtxt(fname)
#f = f_in[0:(len(f_in)//2)]
f = f_in

data_shape = (_x_size, _y_size, _z_size, 3)

# Make the grid
x, y, z = np.meshgrid(np.arange(_x_size),
                      np.arange(_y_size),
                      np.arange(_z_size))

F = f.reshape(data_shape)

print(len(x))
print(len(y))
print(len(z))
print(len(F[:,:,:,0]))
print(len(F[:,:,:,1]))
print(len(F[:,:,:,2]))

ax = plt.figure().add_subplot(projection='3d')

ax.quiver(x, y, z, F[:,:,:,0], F[:,:,:,1], F[:,:,:,2], normalize=True)

plt.show()
