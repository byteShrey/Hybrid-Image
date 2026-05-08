import numpy as np


def build_gaussian_kernel(size, sigma):
    """
    Build a 2D Gaussian kernel from scratch.
    size: odd integer (e.g. 31)
    sigma: standard deviation controlling blur spread
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize so values sum to 1
    kernel /= kernel.sum()
    return kernel
