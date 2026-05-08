import numpy as np


def normalize_image(image):
    """
    Stretch pixel values to [0, 255] range using min-max normalization.
    Works for both grayscale and RGB images.
    """
    img = image.astype(np.float64)
    min_val = img.min()
    max_val = img.max()

    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    normalized = (img - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)


def clip_image(image):
    """
    Clip pixel values to valid [0, 255] range without rescaling.
    Use this when you want to preserve absolute intensity relationships.
    """
    return np.clip(image, 0, 255).astype(np.uint8)
