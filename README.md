# HybridVision
### Frequency-Domain Dual-Perception Image Synthesis

**CS-712: Image Processing — Spring 2026**  
**Team:** Shreyash Kulkarni, Venkata Sesha Sai Eshwar Vudhanthi

---

## What It Does

HybridVision generates hybrid images that appear as two completely different pictures depending on viewing distance. The image seen from far away and the image seen up close are blended using custom low-pass and high-pass filters built entirely from scratch — no OpenCV or image processing libraries used.

---

## Requirements

Python 3.8 or higher is required. Install dependencies with:

```bash
pip install streamlit numpy pillow matplotlib
```

---

## How to Run

1. Clone or download the repository
2. Open a terminal in the project folder
3. Run:

```bash
streamlit run app.py
```

4. The app opens automatically in your browser at `http://localhost:8501`

---

## How to Use

1. **Upload Image A** — the image seen from far away (e.g. a cat face)
2. **Upload Image B** — the image seen up close (e.g. a dog face)
3. Click **Auto-compute parameters** to automatically set the best σ and boost values for your images
4. Fine-tune manually using the three sliders if needed:
   - **σ low** — controls how blurred Image A becomes (higher = A only visible from far)
   - **σ high** — controls how fine Image B's detail is (lower = B only visible up close)
   - **High-freq boost** — amplifies Image B's detail layer
5. Go to the **Sigma Sweep** tab to compare 12 boost variations side by side
6. Click **Download Hybrid Image** in the Result tab to save the output

**To see the illusion:** zoom out your browser with `Ctrl + −` to see Image A, zoom back in to see Image B.

---

## Project Structure

```
hybrid-images/
│
├── app.py                    # Streamlit UI
│
├── algorithms/
│   ├── gaussian.py           # 2D Gaussian kernel (from scratch)
│   ├── convolution.py        # 2D spatial convolution (from scratch)
│   ├── filters.py            # Low-pass and high-pass filters
│   ├── hybrid.py             # Hybrid image creation + auto-parameter computation
│   ├── fft_convolution.py    # FFT-based convolution (from scratch)
│   └── normalize.py          # Image normalization and clipping
│
└── utils/
    └── image_io.py           # Image file I/O (PIL read/write only)
```

---

## Algorithm Overview

1. **Gaussian Kernel** — built from the 2D Gaussian formula, normalized to sum to 1
2. **2D Convolution** — zero-padded spatial convolution implemented with nested loops
3. **FFT Convolution** — faster alternative using the convolution theorem (multiply in frequency domain)
4. **Low-pass Filter** — Gaussian blur applied to Image A, retaining coarse structure
5. **High-pass Filter** — Image B minus its blurred version, retaining fine edges
6. **Hybrid Blend** — `low_pass(A) + high_pass(B) × boost`, clipped to [0, 255]
7. **Auto Parameters** — sigma scaled to image resolution; boost computed by equalising the standard deviation of both frequency layers

---

## Image Selection Tips

For the best hybrid illusion:
- Use two images of the **same subject type** (e.g. two faces, two animals)
- Both images should have the subject **centered and at the same scale**
- Avoid random unrelated photos — the spatial layouts must overlap
