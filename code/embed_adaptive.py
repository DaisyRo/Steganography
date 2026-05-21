import numpy as np
from scipy.ndimage import sobel
from hashlib import sha256

def seed_from_key(key: str) -> int:
    return int(sha256(key.encode('utf-8')).hexdigest()[:16], 16) & 0xFFFFFFFF

def local_gradient_map(img):
    gx = sobel(img.astype(np.float32), axis=1)
    gy = sobel(img.astype(np.float32), axis=0)
    return np.hypot(gx, gy)

def embed_adaptive(img_arr, bitstream, k=1, capacity_fraction=0.5, key=None):
    h, w = img_arr.shape
    capacity = h * w
    S = local_gradient_map(img_arr)
    flatS = S.flatten()
    indices = np.argsort(flatS)[::-1]  # по убыванию градиента
    n_to_write = min(len(bitstream), int(capacity * capacity_fraction))
    selected = indices[:n_to_write]
    if key is not None:
        rng = np.random.RandomState(seed_from_key(key))
        rng.shuffle(selected)
    flat = img_arr.flatten().copy()
    shift = k - 1
    mask = ~(1 << shift) & 0xFF
    for i, pos in enumerate(selected):
        flat[pos] = (flat[pos] & mask) | (bitstream[i] << shift)
    return flat.reshape((h, w)).astype(np.uint8), n_to_write, selected

def extract_adaptive(stego_arr, selected_indices, k=1, n_bits=None):
    flat = stego_arr.flatten()
    shift = k - 1
    if n_bits is None:
        n_bits = len(selected_indices)
    bits = [int((flat[pos] >> shift) & 1) for pos in selected_indices[:n_bits]]
    return bits
