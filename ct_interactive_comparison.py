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

ct1, labels1, masks1 = load_and_prepare("1a.npy")
ct2, labels2, masks2 = load_and_prepare("1b.npy")  # second file

cmap = get_cmap("tab10")

pl = pv.Plotter(shape=(1, 2), border=False, window_size=(1600, 800))
pl.link_views()

pl.subplot(0, 0)
pl.add_volume(ct1, cmap="bone", opacity="sigmoid_9", shade=False)
for i, mask_idx in enumerate(masks1):
    binary_mask = (labels1 == mask_idx).astype(np.uint8)
    mask_image = pv.ImageData(dimensions=binary_mask.shape)
    mask_image.spacing = (1.0, 1.0, 1.0)
    mask_image.point_data["mask"] = binary_mask.ravel(order="F")
    contour = mask_image.contour(isosurfaces=[0.5], scalars="mask")
    color = cmap(i % 10)[:3]
    pl.add_mesh(contour, color=color, opacity=0.4, name=f"Organ_L_{mask_idx}", show_scalar_bar=False)
pl.add_text("Left Volume", font_size=10)

pl.subplot(0, 1)
pl.add_volume(ct2, cmap="bone", opacity="sigmoid_9", shade=False)
for i, mask_idx in enumerate(masks2):
    binary_mask = (labels2 == mask_idx).astype(np.uint8)
    mask_image = pv.ImageData(dimensions=binary_mask.shape)
    mask_image.spacing = (1.0, 1.0, 1.0)
    mask_image.point_data["mask"] = binary_mask.ravel(order="F")
    contour = mask_image.contour(isosurfaces=[0.5], scalars="mask")
    color = cmap(i % 10)[:3]
    pl.add_mesh(contour, color=color, opacity=0.4, name=f"Organ_R_{mask_idx}", show_scalar_bar=False)
pl.add_text("Right Volume", font_size=10)

pl.add_axes()
pl.show()
