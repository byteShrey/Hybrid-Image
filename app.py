import streamlit as st
import numpy as np

from utils.image_io import load_image, save_image_bytes, resize_to_match
from algorithms.hybrid import create_hybrid, sigma_sweep

st.set_page_config(page_title="HybridVision", layout="wide")
st.title("HybridVision")
st.caption("Frequency-Domain Dual-Perception Image Synthesis — CS-712 Term Project")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Images")
    file_a = st.file_uploader("Image A  —  seen from far away", type=["jpg", "jpeg", "png"])
    file_b = st.file_uploader("Image B  —  seen up close",      type=["jpg", "jpeg", "png"])

    st.divider()
    st.header("Parameters")
    sigma_low  = st.slider("σ low",           1.0, 60.0, 10.0, step=0.5)
    sigma_high = st.slider("σ high",          1.0, 30.0, 10.0, step=0.5)
    high_boost = st.slider("High-freq boost", 0.5,  3.0,  1.0, step=0.1)
    grayscale  = st.checkbox("Grayscale mode", value=False)

    st.divider()
    st.header("Convolution Method")
    method_label = st.radio(
        "Choose path",
        ["FFT-based (fast)", "Spatial (slow)"],
        index=0,
    )
    conv_method = "fft" if method_label.startswith("FFT") else "spatial"

if not file_a or not file_b:
    st.info("Upload both images in the sidebar to get started.")
    st.stop()

# ── Load & process ────────────────────────────────────────────────────────────
img_a = load_image(file_a, grayscale=grayscale)
img_b = load_image(file_b, grayscale=grayscale)
img_a, img_b = resize_to_match(img_a, img_b)

with st.spinner("Computing…"):
    hybrid, low, high_display = create_hybrid(
        img_a, img_b, sigma_low, sigma_high, method=conv_method, high_boost=high_boost
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Hybrid Result", "Sigma Sweep"])

# ── Tab 1: Hybrid Result ──────────────────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Image A")
        st.image(img_a.astype(np.uint8), use_container_width=True)
    with c2:
        st.subheader("Image B")
        st.image(img_b.astype(np.uint8), use_container_width=True)
    with c3:
        st.subheader("Hybrid")
        st.image(hybrid, use_container_width=True)

    st.divider()
    c4, c5 = st.columns(2)
    with c4:
        st.subheader(f"Low-pass of A  (σ = {sigma_low})")
        st.image(low, use_container_width=True)
    with c5:
        st.subheader(f"High-pass of B  (σ = {sigma_high})")
        st.image(high_display, use_container_width=True)

    st.download_button(
        "Download Hybrid Image",
        data=save_image_bytes(hybrid),
        file_name="hybrid_result.png",
        mime="image/png",
    )

# ── Tab 2: Sigma Sweep ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Sigma Sweep")

    boost_values = [round(b, 1) for b in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]
    fixed_sl, fixed_sh = 10, 10

    with st.spinner("Running sweep…"):
        sweep_results = []
        for hb in boost_values:
            result, _, _ = create_hybrid(img_a, img_b, fixed_sl, fixed_sh, method=conv_method, high_boost=hb)
            sweep_results.append((hb, result))

    # Display in rows of 4
    for row_start in range(0, len(sweep_results), 4):
        row = sweep_results[row_start:row_start + 4]
        cols = st.columns(4)
        for col, (hb, result) in zip(cols, row):
            with col:
                st.image(result, use_container_width=True)
                st.caption(f"σ low = {fixed_sl}  \nσ high = {fixed_sh}  \nboost = {hb}")
