import numpy as np
import pyvista as pv
from matplotlib.cm import get_cmap

data = np.load("volume_mask2.npy")
ct = data[..., 0].astype(np.float32)
labels = data[..., 1].astype(np.uint8)

ct = np.transpose(ct, (2, 1, 0))
labels = np.transpose(labels, (2, 1, 0))

unique_labels = np.unique(labels)
present_masks = unique_labels[unique_labels > 0]
print(f"Found masks: {present_masks}")

cmap = get_cmap("tab10")

pl = pv.Plotter()
pl.add_volume(ct, cmap="bone", opacity="sigmoid_9", shade=False)

for i, mask_idx in enumerate(present_masks):
    binary_mask = (labels == mask_idx).astype(np.uint8)
    dims = binary_mask.shape
    mask_image = pv.ImageData(dimensions=dims)
    mask_image.spacing = (1.0, 1.0, 1.0)
    mask_image.point_data["mask"] = binary_mask.ravel(order="F")
    contour = mask_image.contour(isosurfaces=[0.5], scalars="mask")
    color = cmap(i % 10)[:3]
    pl.add_mesh(contour, color=color, opacity=0.4, name=f"Organ_{mask_idx}", show_scalar_bar=False)

pl.add_axes()
pl.add_text("CT + Organ Masks", font_size=12)
pl.show()
