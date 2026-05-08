import numpy as np


def _next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fft_convolve2d(image, kernel):
    """
    2D convolution via the convolution theorem: multiply in frequency domain,
    then inverse-transform. Equivalent to spatial convolution but much faster
    for large kernels.
    image: 2D numpy array (H x W)
    kernel: 2D numpy array (k x k)
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Full linear convolution output size
    out_h = img_h + k_h - 1
    out_w = img_w + k_w - 1

    # Pad to next power of 2 for FFT efficiency
    fft_h = _next_power_of_2(out_h)
    fft_w = _next_power_of_2(out_w)

    # Zero-pad both image and kernel to the same FFT grid
    img_padded = np.zeros((fft_h, fft_w), dtype=np.float64)
    img_padded[:img_h, :img_w] = image

    ker_padded = np.zeros((fft_h, fft_w), dtype=np.float64)
    ker_padded[:k_h, :k_w] = kernel

    # Convolution theorem: pointwise multiply in frequency domain
    result = np.real(np.fft.ifft2(np.fft.fft2(img_padded) * np.fft.fft2(ker_padded)))

    # Extract valid region (same size as input image)
    pad_h = k_h // 2
    pad_w = k_w // 2
    return result[pad_h:pad_h + img_h, pad_w:pad_w + img_w]


def fft_convolve2d_rgb(image, kernel):
    """Apply FFT-based 2D convolution independently to each RGB channel."""
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        result[:, :, c] = fft_convolve2d(image[:, :, c], kernel)
    return result


def compute_magnitude_spectrum(image):
    """
    Log-scaled magnitude spectrum of an image, centered at zero frequency.
    Used to visualize which frequency components each filter preserves.
    Returns a uint8 grayscale array in [0, 255].
    """
    gray = image.mean(axis=2) if image.ndim == 3 else image.copy()
    F_shifted = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(F_shifted))
    min_val, max_val = magnitude.min(), magnitude.max()
    if max_val > min_val:
        magnitude = (magnitude - min_val) / (max_val - min_val) * 255.0
    return magnitude.astype(np.uint8)
