import os, struct, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

DATASETS = {'BOSS': 'data/bmp_boss', 'MED': 'data/bmp_med', 'OTHER': 'data/bmp_other'}
SECRET = 'code/secret.bin'
OUT = 'results/research'

def load(p): return np.array(Image.open(p).convert('L'), dtype=np.uint8)
def embed(arr, k, bits):
    f = arr.flatten().copy(); m = 1<<(k-1); iv = 0xFF^m
    for i in range(min(len(bits), f.size)): f[i] = (f[i]&iv)|(bits[i]<<(k-1))
    return f.reshape(arr.shape)
def metrics(o, s):
    mse = np.mean((o-s)**2)
    psnr = 10*np.log10(255**2/mse) if mse>0 else 100.0
    return mse, psnr, ssim(o, s, data_range=255)

with open(SECRET,'rb') as f: msg=f.read()
bits = [(b>>i)&1 for b in (struct.pack('>I',len(msg))+msg) for i in range(7,-1,-1)]

print(f"{'Набор':<8} | {'k':<2} | {'MSE':<7} | {'PSNR':<7} | {'SSIM':<7} | {'ΔMSE':<5} | {'ΔPSNR':<6}")
print("-"*62)
base_mse = {}
for ds, path in DATASETS.items():
    fname = sorted([f for f in os.listdir(path) if f.endswith('.bmp')])[0]
    orig = load(os.path.join(path, fname))
    for k in [1,2,3]:
        st = embed(orig, k, bits)
        mse, psnr, ss = metrics(orig, st)
        if k==1: base_mse[ds] = mse
        dmse = mse / base_mse[ds] if base_mse[ds]>0 else 1.0
        dpsnr = 51.25 - psnr  # отклонение от теоретического максимума
        print(f"{ds:<8} | {k:<2} | {mse:<7.4f} | {psnr:<7.2f} | {ss:<7.4f} | {dmse:<5.2f}x | {dpsnr:<6.2f}dB")
print("-"*62)
print("✅ Таблица готова")
