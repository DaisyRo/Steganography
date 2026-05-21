import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATASETS = {'BOSS': 'data/bmp_boss', 'MED': 'data/bmp_med', 'OTHER': 'data/bmp_other'}
STEGO_DIR = 'results/research'
OUT_DIR = 'results/research'
os.makedirs(OUT_DIR, exist_ok=True)

def load(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)

def plot_histogram(orig_path, stego_path, out_path, title):
    orig = load(orig_path)
    stego = load(stego_path)
    
    plt.figure(figsize=(8, 5))
    plt.hist(orig.flatten(), bins=64, alpha=0.6, label='Original', color='#1f77b4', edgecolor='black')
    plt.hist(stego.flatten(), bins=64, alpha=0.6, label='Stego (k=1)', color='#ff7f0e', edgecolor='black')
    plt.title(title)
    plt.xlabel('Intensity (0–255)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ {out_path}")

print("📊 Построение гистограмм (пункт d)...")
for ds, path in DATASETS.items():
    fname = sorted([f for f in os.listdir(path) if f.endswith('.bmp')])[0]
    orig_path = os.path.join(path, fname)
    base = os.path.splitext(fname)[0]
    stego_name = f"{ds}_{base}_k1.bmp"
    stego_path = os.path.join(STEGO_DIR, stego_name)
    
    if os.path.exists(stego_path):
        out_name = f"{ds}_{base}_hist_k1.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plot_histogram(orig_path, stego_path, out_path, f"Histogram: {ds} — {fname}")

print("✅ Гистограммы сохранены в results/research/")

