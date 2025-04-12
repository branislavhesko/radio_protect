import numpy as np
import pyvista as pv

# --- Load the 4D array (Z, Y, X, 2) ---
data = np.load("volume_mask.npy")

# --- Extract CT and label map ---
ct = data[..., 0].astype(np.float32)
labels = data[..., 1].astype(np.uint8)

# --- Transpose to (X, Y, Z) ---
ct = np.transpose(ct, (2, 1, 0))
labels = np.transpose(labels, (2, 1, 0))

# --- Detect present masks (1–6) ---
unique_labels = np.unique(labels)
present_masks = unique_labels[unique_labels > 0]
print(f"Found masks: {present_masks}")

# --- Color palette for masks ---
from matplotlib.cm import get_cmap
cmap = get_cmap("tab10")  # Up to 10 distinct colors

# --- Create PyVista volume from CT ---
pl = pv.Plotter()
pl.add_volume(ct, cmap="bone", opacity="sigmoid_9", shade=False)

# --- Add all present masks as contours ---
for i, mask_idx in enumerate(present_masks):
    # Extract binary mask
    binary_mask = (labels == mask_idx).astype(np.uint8)

    # Wrap as PyVista ImageData
    dims = binary_mask.shape
    mask_image = pv.ImageData(dimensions=dims)
    mask_image.spacing = (1.0, 1.0, 1.0)
    mask_image.point_data["mask"] = binary_mask.ravel(order="F")

    # Generate surface
    contour = mask_image.contour(isosurfaces=[0.5], scalars="mask")

    # Pick a color from the colormap
    color = cmap(i % 10)[:3]  # Get RGB

    # Add mesh to plot
    pl.add_mesh(
        contour,
        color=color,
        opacity=0.4,
        name=f"Organ_{mask_idx}",
        show_scalar_bar=False
    )

pl.add_axes()
pl.add_text("CT + Organ Masks", font_size=12)
pl.show()
