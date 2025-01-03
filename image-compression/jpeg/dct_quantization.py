import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    """Compute the 2D DCT of an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Compute the 2D inverse DCT of an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def compute_dct_blocks(image):
    """Compute the 2D DCT for all 8x8 blocks of an image."""
    h, w = image.shape
    block_size = 8
    dct_blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_blocks.append(dct2(block))

    return dct_blocks

def quantize_block(block, quantization_matrix):
    """Quantize a single 8x8 block using the quantization matrix."""
    return np.round(block / quantization_matrix)

def dequantize_block(block, quantization_matrix):
    """Dequantize a single 8x8 block using the quantization matrix."""
    return block * quantization_matrix