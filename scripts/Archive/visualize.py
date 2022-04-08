import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

#PLOT_TITLE = "Mesh Potentials (V): Threads = {}, Size = {} cm"
PLOT_TITLE = "Mesh Potentials (V): Size = {} cm"

def plot(x, y, v, threads, size):
    plt. clf()
    
    #plt.title(PLOT_TITLE.format(threads, size))

    plt.title(PLOT_TITLE.format(size))

    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)
    V=v.reshape(len(y),len(x))

    mesh = plt.pcolormesh(X,Y,V, cmap='jet')
    plt.colorbar(mesh)

    # [plt.hlines(y=yVal, xmin=0.0, xmax=65.0, linewidth=0.5) for yVal in np.arange(0, 50, float(size))]
    # [plt.vlines(x=xVal, ymin=0.0, ymax=50.0, linewidth=0.5) for xVal in np.arange(0, 65, float(size))]

    plt.savefig("output/heatmaps/mesh_{}_{}.png".format(threads, size))


if not os.path.exists('output/heatmaps'):
    os.makedirs('output/heatmaps')

for fname in glob.glob("output/mesh_*.out"):
    # Extract relevant file info from filename
    results = re.search("output/mesh_(\d*).out", fname)
    size = results.group(1)
    # Import file contents
    
    #data = np.genfromtxt(fname,delimiter='\t')
    with open("randomfile.csv") as file_name:
        array = np.loadtxt(file_name, delimiter=",")


    for i in reader:
        x = i % _x_size; // TODO:CONVERT i -> x
        y = (i / _x_size) % _y_size; // TODO:CONVERT i -> y
        z = (i / _x_size) / _y_size; // TODO:CONVERT i -> z
    
    x=data[:,0]
    y=data[:,1]
    z=
    v=data[:,2]

    # Export the plot to file
#    plot(x, y, v, threads, size)
