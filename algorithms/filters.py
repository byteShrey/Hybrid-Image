import numpy as np
from algorithms.gaussian import build_gaussian_kernel
from algorithms.convolution import convolve2d, convolve2d_rgb


def low_pass_filter(image, sigma, kernel_size=None):
    """
    Blur image using a Gaussian low-pass filter (keeps low frequencies = coarse structure).
    sigma: controls amount of blur — higher = more blur
    """
    if kernel_size is None:
        # Rule of thumb: kernel covers ±3 sigma
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    kernel = build_gaussian_kernel(kernel_size, sigma)

    if image.ndim == 3:
        return convolve2d_rgb(image.astype(np.float64), kernel)
    else:
        return convolve2d(image.astype(np.float64), kernel)


def high_pass_filter(image, sigma, kernel_size=None):
    """
    Extract high-frequency detail by subtracting the low-pass version.
    Result contains edges and fine texture (what you see up close).
    """
    blurred = low_pass_filter(image, sigma, kernel_size)
    return image.astype(np.float64) - blurred
