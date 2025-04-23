import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

st.title("🎨 Color Replacer")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Color pickers
st.subheader("Select the color to be replaced (Target)")
target_r = st.slider("Red (Target)", 0, 255, 0)
target_g = st.slider("Green (Target)", 0, 255, 0)
target_b = st.slider("Blue (Target)", 0, 255, 255)

st.subheader("Select the replacement color")
replace_r = st.slider("Red (Replacement)", 0, 255, 0)
replace_g = st.slider("Green (Replacement)", 0, 255, 255)
replace_b = st.slider("Blue (Replacement)", 0, 255, 0)

threshold = st.slider("Color Matching Threshold", 0, 100, 40)

if uploaded_file:
    # Read and convert image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    if image_np.shape[-1] == 4:  # remove alpha if present
        image_np = image_np[:, :, :3]
    
    target_color = np.array([target_r, target_g, target_b])
    replacement_color = np.array([replace_r, replace_g, replace_b])
    
    # Convert RGB to OpenCV format (BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Create mask
    lower = np.clip(target_color - threshold, 0, 255)
    upper = np.clip(target_color + threshold, 0, 255)
    mask = cv2.inRange(image_rgb, lower, upper)

    # Apply replacement
    result_image = image_rgb.copy()
    result_image[mask > 0] = replacement_color

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    st.subheader("Modified Image")
    st.image(result_image, use_container_width=True)

    # Download option
    result_pil = Image.fromarray(result_image)
    buffer = BytesIO()
    result_pil.save(buffer, format="PNG")
    byte_img = buffer.getvalue()

    st.download_button("📥 Download Modified Image", data=byte_img, file_name="modified_image.png", mime="image/png")
