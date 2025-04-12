import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Load the 4D data ---
data = np.load("volume_mask.npy")  # shape: (Z, Y, X, 7)

ct = data[..., 0].astype(np.float32)  # Base CT
num_masks = data.shape[-1] - 1        # Masks = channels 1 to 6

print(data.shape[-1])

# --- Select which mask to overlay ---
if num_masks > 1:
    mask_idx = st.slider("Organ Mask (1–6)", 1, num_masks, 1)
else:
    st.info("Only one mask available.")
    mask_idx = 1

mask = data[..., mask_idx]

# --- Orientation selection ---
orientation = st.radio("Orientation", ["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"])

# --- Determine slice axis & range ---
if orientation == "Axial (XY)":
    max_index = ct.shape[0]
    axis = 0
elif orientation == "Coronal (XZ)":
    max_index = ct.shape[1]
    axis = 1
else:
    max_index = ct.shape[2]
    axis = 2

index = st.slider("Slice Index", 0, max_index - 1, max_index // 2)

# --- Extract slices ---
if axis == 0:
    ct_slice = ct[index, :, :]
    mask_slice = mask[index, :, :]
elif axis == 1:
    ct_slice = ct[:, index, :]
    mask_slice = mask[:, index, :]
else:
    ct_slice = ct[:, :, index]
    mask_slice = mask[:, :, index]

# --- Display options ---
colormap = st.selectbox("CT Colormap", ["gray", "bone", "viridis", "plasma", "magma", "inferno", "hot"])
show_mask = st.checkbox("Show Mask Overlay", value=True)
alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.4)

# --- CT brightness/contrast ---
default_vmin = float(np.percentile(ct_slice, 1))
default_vmax = float(np.percentile(ct_slice, 99))
vmin = st.slider("Window Min (brightness)", float(ct_slice.min()), float(ct_slice.max()), default_vmin)
vmax = st.slider("Window Max (contrast)", float(ct_slice.min()), float(ct_slice.max()), default_vmax)

# --- Plot ---
fig, ax = plt.subplots()
ax.imshow(ct_slice.T, cmap=colormap, origin="lower", vmin=vmin, vmax=vmax)

if show_mask:
    from matplotlib.colors import ListedColormap
    # Create a binary mask: pixels where mask_slice is nonzero
    mask_binary = (mask_slice.T > 0).astype(int)
    # Create a custom colormap: index 0 = transparent, index 1 = bright red with the selected alpha
    bright_red = ListedColormap([[1, 0, 0, 0], [1, 0, 0, alpha]])
    ax.imshow(
        mask_binary,
        cmap=bright_red,
        origin="lower",
        interpolation="none"
    )

ax.set_title(f"{orientation} Slice {index} — Mask {mask_idx}")
ax.axis("off")
st.pyplot(fig)
