import numpy as np


def convolve2d(image, kernel):
    """
    2D convolution implemented from scratch using zero-padding.
    Works on a single-channel (grayscale) image.
    image: 2D numpy array (H x W)
    kernel: 2D numpy array (k x k), must be square and odd-sized
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Zero-pad the image
    padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w), dtype=np.float64)
    padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output


def convolve2d_rgb(image, kernel):
    """
    Apply 2D convolution independently to each RGB channel.
    image: 3D numpy array (H x W x 3)
    kernel: 2D numpy array
    """
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d(image[:, :, c], kernel)
    return result
