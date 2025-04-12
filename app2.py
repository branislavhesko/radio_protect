import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

data = np.load("volume_mask.npy")  # shape: (Z, Y, X, 7)
num_channels = data.shape[-1]

channel = st.slider("Channel", 0, num_channels - 1, 1)
volume = data[..., channel].astype(np.float32)

# --- Orientation selection ---
orientation = st.radio("Orientation", ["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"])

# --- Determine slice range based on orientation ---
if orientation == "Axial (XY)":
    max_index = volume.shape[0]
    axis = 0
elif orientation == "Coronal (XZ)":
    max_index = volume.shape[1]
    axis = 1
else:  # Sagittal (YZ)
    max_index = volume.shape[2]
    axis = 2

index = st.slider("Slice Index", 0, max_index - 1, max_index // 2)

# --- Extract slice ---
if axis == 0:
    slice_ = volume[index, :, :]
elif axis == 1:
    slice_ = volume[:, index, :]
else:
    slice_ = volume[:, :, index]

# --- Colormap selection ---
colormap = st.selectbox("Colormap", ["gray", "bone", "viridis", "plasma", "magma", "inferno", "hot"])

# --- Window/level controls ---
slice_min = float(slice_.min())
slice_max = float(slice_.max())

# Handle edge case: min == max
if slice_min == slice_max:
    slice_max = slice_min + 1.0

default_vmin = float(np.percentile(slice_, 1))
default_vmax = float(np.percentile(slice_, 99))

vmin = st.slider("Window Min (brightness)", slice_min, slice_max, default_vmin)
vmax = st.slider("Window Max (contrast)", slice_min, slice_max, default_vmax)

# --- Plot using matplotlib ---
fig, ax = plt.subplots()
ax.imshow(slice_.T, cmap=colormap, origin="lower", vmin=vmin, vmax=vmax)
ax.set_title(f"{orientation} Slice {index} Channel {channel}")
ax.axis("off")

st.pyplot(fig)
