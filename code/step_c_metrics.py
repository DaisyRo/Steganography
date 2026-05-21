import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import struct

DATASETS = {'BOSS': 'data/bmp_boss', 'MED': 'data/bmp_med', 'OTHER': 'data/bmp_other'}
SECRET = 'code/secret.bin'
OUT_DIR = 'results/research'
os.makedirs(OUT_DIR, exist_ok=True)

def load(path): return np.array(Image.open(path).convert('L'), dtype=np.uint8)
def save(arr, path): Image.fromarray(arr.astype(np.uint8)).save(path, format='BMP')

def embed(arr, k, msg_bits):
    flat = arr.flatten().copy()
    mask = 1 << (k - 1)
    inv = 0xFF ^ mask
    for i in range(min(len(msg_bits), flat.size)):
        flat[i] = (flat[i] & inv) | (msg_bits[i] << (k - 1))
    return flat.reshape(arr.shape)

def calc_metrics(orig, stego):
    mse = np.mean((orig - stego) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100.0
    ss = ssim(orig, stego, data_range=255)
    return round(mse, 4), round(psnr, 4), round(ss, 6)

# Подготовка битов сообщения (32-битный заголовок длины + данные)
with open(SECRET, 'rb') as f: msg = f.read()
header = struct.pack('>I', len(msg))
full = header + msg
bits = [(b >> i) & 1 for b in full for i in range(7, -1, -1)]

print(f"{'Набор':<8} | {'k':<2} | {'MSE':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Файл'}")
print("-" * 88)

for ds, path in DATASETS.items():
    fname = sorted([f for f in os.listdir(path) if f.endswith('.bmp')])[0]
    orig = load(os.path.join(path, fname))
    base = os.path.splitext(fname)[0]

    for k in [1, 2, 3]:
        stego = embed(orig, k, bits)
        out_name = f"{ds}_{base}_k{k}.bmp"
        out_path = os.path.join(OUT_DIR, out_name)
        save(stego, out_path)

        mse, psnr, ss = calc_metrics(orig, stego)
        print(f"{ds:<8} | {k:<2} | {mse:<8.4f} | {psnr:<10.2f} | {ss:<8.4f} | {out_name}")

print("-" * 88)
print("✅ Все 9 стего-файлов сохранены в results/research/")
