import numpy as np

from algorithms.gaussian import build_gaussian_kernel
from algorithms.filters import low_pass_filter, high_pass_filter
from algorithms.fft_convolution import fft_convolve2d, fft_convolve2d_rgb
from algorithms.normalize import clip_image


def _low_pass_fft(image, sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    kernel = build_gaussian_kernel(kernel_size, sigma)
    if image.ndim == 3:
        return fft_convolve2d_rgb(image.astype(np.float64), kernel)
    return fft_convolve2d(image.astype(np.float64), kernel)


def _high_pass_fft(image, sigma, kernel_size=None):
    blurred = _low_pass_fft(image, sigma, kernel_size)
    return image.astype(np.float64) - blurred


def create_hybrid(image_a, image_b, sigma_low, sigma_high, method='fft', high_boost=1.5):
    """
    Blend low frequencies from image_a with high frequencies from image_b.

    method='fft'     — uses FFT-based convolution (fast, recommended)
    method='spatial' — uses the from-scratch spatial convolution in convolution.py
    high_boost       — multiplier on the high-frequency layer; increase to make
                       the close-up image (B) more dominant up close.

    Returns:
        hybrid        — final blended image, uint8 clipped to [0, 255]
        low           — low-pass of image_a, uint8
        high_display  — high-pass of image_b shifted by +128 for display, uint8
    """
    if method == 'fft':
        low = _low_pass_fft(image_a, sigma_low)
        high = _high_pass_fft(image_b, sigma_high)
    else:
        low = low_pass_filter(image_a, sigma_low)
        high = high_pass_filter(image_b, sigma_high)

    hybrid = low + high * high_boost

    return (
        clip_image(hybrid),
        clip_image(low),
        clip_image(high + 128.0),   # shift so high-pass is visible (zero → gray)
    )


def compute_auto_params(image_a, image_b, method='fft'):
    """
    Automatically compute sigma and boost for a given image pair.

    Sigma is scaled so it behaves like sigma=10 on an 800px image,
    regardless of the actual image resolution.

    Boost is computed by equalising the energy (std dev) of the low-pass
    and high-pass layers so both contribute equally to the hybrid.

    Returns (sigma_low, sigma_high, boost) as floats.
    """
    h, w = image_a.shape[:2]
    ref = min(w, h)
    sigma = float(max(5, round(10 * ref / 800)))

    low  = _low_pass_fft(image_a, sigma)  if method == 'fft' else low_pass_filter(image_a, sigma)
    high = _high_pass_fft(image_b, sigma) if method == 'fft' else high_pass_filter(image_b, sigma)

    std_low  = float(np.std(low))
    std_high = float(np.std(high))

    boost = round(std_low / std_high, 1) if std_high > 1e-8 else 1.0
    boost = float(np.clip(boost, 0.5, 3.0))

    return sigma, sigma, boost


def sigma_sweep(image_a, image_b, sigma_values, method='fft'):
    """
    Generate hybrid images for each sigma in sigma_values (same value used for
    both sigma_low and sigma_high). Returns list of (sigma, hybrid_uint8) tuples.
    """
    return [
        (sigma, create_hybrid(image_a, image_b, sigma, sigma, method)[0])
        for sigma in sigma_values
    ]
