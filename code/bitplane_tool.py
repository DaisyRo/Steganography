from PIL import Image
import numpy as np
from hashlib import sha256

def load_image(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)

def save_image(arr, path):
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)

def bits_from_bytes(bts):
    bits = []
    for byte in bts:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def bytes_from_bits(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Длина битовой последовательности должна быть кратна 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)

def pack_payload(logo_bytes):
    MAGIC = b'WTR1'
    nbits = len(logo_bytes) * 8
    header = MAGIC + nbits.to_bytes(4, 'big')
    digest = sha256(logo_bytes).digest()
    return header + digest + logo_bytes

def unpack_payload(payload_bytes):
    if len(payload_bytes) < 4 + 4 + 32:
        return False, None
    MAGIC = payload_bytes[:4]
    if MAGIC != b'WTR1':
        return False, None
    nbits = int.from_bytes(payload_bytes[4:8], 'big')
    digest = payload_bytes[8:40]
    payload = payload_bytes[40:]
    if len(payload) * 8 < nbits:
        return False, None
    payload = payload[:(nbits + 7)//8]
    if sha256(payload).digest() != digest:
        return False, None
    return True, payload
