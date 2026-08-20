"""Official demosaicing module for the 16-band snapshot spectral camera."""

import numpy as np
from PIL import Image


def X2Cube(img: np.ndarray, cellSize: int = 4) -> np.ndarray:
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)


def load_cube(path, cell_size: int = 4) -> np.ndarray:
    img = np.array(Image.open(path))
    cube = X2Cube(img, cellSize=cell_size)
    return cube.astype(np.float32)


def normalize_cube(cube: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    max_val = float(2 ** bit_depth - 1)
    return np.clip(cube / max_val, 0.0, 1.0)