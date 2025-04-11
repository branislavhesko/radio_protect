import numpy as np
from vedo import Volume, dataurl
from vedo.applications import RayCastPlotter

# Load Volume data
volume = np.load("volume.npy")
embryo = Volume(volume, spacing=(1, 1, 1))
embryo.mode(1).cmap("jet")  # change visual properties

# Create a Plotter instance and show
plt = RayCastPlotter(embryo, bg='black', bg2='blackboard', axes=7)
plt.show(viewup="z")
plt.close()