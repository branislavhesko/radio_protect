import numpy as np
import pyvista as pv
from matplotlib.cm import get_cmap

def load_and_prepare(filepath):
    data = np.load(filepath)
    ct = data[..., 0].astype(np.float32)
    labels = data[..., 1].astype(np.uint8)
    ct = np.transpose(ct, (2, 1, 0))
    labels = np.transpose(labels, (2, 1, 0))
    present_masks = np.unique(labels)
    present_masks = present_masks[present_masks > 0]
    return ct, labels, present_masks

# Load both volumes
ct1, labels1, masks1 = load_and_prepare("1a.npy")
ct2, labels2, masks2 = load_and_prepare("1b.npy")

# --- Compute the difference between CT2 and CT1 ---
# Here, we subtract CT1 from CT2. You can change the order depending on your needs.
diff = ct2 - ct1
abs_diff = np.abs(diff)

# Define a threshold: Here we use the 95th percentile of the absolute differences.
# This means only the top 5% of voxel changes are considered significant.
threshold = np.percentile(abs_diff, 95)
print("Threshold for significant difference:", threshold)

# Create a new volume for differences:
# Only values where the absolute difference exceeds the threshold are kept;
# the rest are set to zero.
diff_vol = diff.copy()
diff_vol[abs_diff < threshold] = 0

# Determine the maximum absolute difference for proper colormap scaling.
# This centers the diverging colormap (coolwarm) at zero.
abs_max = np.max(np.abs(diff_vol))
print("Maximum absolute difference for coloring:", abs_max)

# --- Visualization setup using PyVista ---
# Define colormaps: 
# 'bone' for CT1 and 'coolwarm' (a diverging colormap) for the differences.
base_cmap = "bone"
diff_cmap = "coolwarm"

# Create a PyVista plotter with a single view.
pl = pv.Plotter(border=False, window_size=(800, 800))

# Render the base CT1 volume.
pl.add_volume(ct1, cmap=base_cmap, opacity="sigmoid_9", shade=False, name="CT1 Base")

# Overlay the significant differences if any exist.
# The 'clim' parameter sets the scalar range from -abs_max to abs_max,
# ensuring that 0 maps to the center of the diverging colormap.
if abs_max > 0:
    pl.add_volume(diff_vol, cmap=diff_cmap, opacity="linear",
                  clim=(-abs_max, abs_max), shade=False, name="Differences")

pl.add_text("CT1 with Highlighted Significant Differences", font_size=10)
pl.add_axes()
pl.show()
