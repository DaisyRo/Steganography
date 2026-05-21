import numpy as np
from hashlib import sha256

def seed_from_key(key: str) -> int:
    return int(sha256(key.encode('utf-8')).hexdigest()[:16], 16) & 0xFFFFFFFF

def embed_lsb(img_arr, bitstream, k=1, key='secret', capacity_fraction=0.5):
    h, w = img_arr.shape
    capacity = h * w
    n_to_write = min(len(bitstream), int(capacity * capacity_fraction))
    rng = np.random.RandomState(seed_from_key(key))
    perm = rng.permutation(capacity)
    selected = perm[:n_to_write]
    flat = img_arr.flatten().copy()
    shift = k - 1
    mask = ~(1 << shift) & 0xFF
    for i, pos in enumerate(selected):
        flat[pos] = (flat[pos] & mask) | (bitstream[i] << shift)
    return flat.reshape((h, w)).astype(np.uint8), n_to_write, selected

def extract_lsb(stego_arr, k=1, key='secret', n_bits=None):
    h, w = stego_arr.shape
    capacity = h * w
    rng = np.random.RandomState(seed_from_key(key))
    perm = rng.permutation(capacity)
    if n_bits is None:
        n_bits = capacity
    selected = perm[:n_bits]
    flat = stego_arr.flatten()
    shift = k - 1
    bits = [int((flat[pos] >> shift) & 1) for pos in selected]
    return bits
