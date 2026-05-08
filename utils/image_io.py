import io
import numpy as np
from PIL import Image


def load_image(source, grayscale=False):
    """
    Load an image from a file path or a Streamlit UploadedFile object.
    Returns a float64 numpy array with pixel values in [0, 255].
    """
    img = Image.open(source)
    img = img.convert('L' if grayscale else 'RGB')
    return np.array(img, dtype=np.float64)


def save_image_bytes(array):
    """
    Encode a uint8 numpy array as PNG bytes (for Streamlit download buttons).
    """
    img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def resize_to_match(image_a, image_b):
    """
    Resize both images to the same (min) dimensions using nearest-neighbor
    sampling implemented with NumPy index arithmetic — no PIL processing used.
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    target_h, target_w = min(h_a, h_b), min(w_a, w_b)
    return _nn_resize(image_a, target_h, target_w), _nn_resize(image_b, target_h, target_w)


def _nn_resize(image, new_h, new_w):
    """Nearest-neighbor resize via NumPy row/column index mapping."""
    h, w = image.shape[:2]
    if h == new_h and w == new_w:
        return image
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    return image[row_idx][:, col_idx]
