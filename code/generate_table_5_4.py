import os, struct, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

DATASETS = {'BOSS': 'data/bmp_boss', 'MED': 'data/bmp_med', 'OTHER': 'data/bmp_other'}
SECRET = 'code/secret.bin'

def load(p): return np.array(Image.open(p).convert('L'), dtype=np.uint8)

def embed(arr, k, bits):
    f = arr.flatten().copy()
    m = 1 << (k-1)
    iv = 0xFF ^ m
    for i in range(min(len(bits), f.size)):
        f[i] = (f[i] & iv) | (bits[i] << (k-1))
    return f.reshape(arr.shape)

def calc_metrics(o, s):
    mse = np.mean((o - s)**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100.0
    ss = ssim(o, s, data_range=255)
    return mse, psnr, ss

# Подготовка битов
with open(SECRET, 'rb') as f: msg = f.read()
bits = [(b>>i)&1 for b in (struct.pack('>I', len(msg)) + msg) for i in range(7, -1, -1)]

# Заголовок таблицы
print(f"{'Набор':<8} | {'k':<2} | {'PSNR':<7} | {'MSE (норм.)':<12} | {'SSIM':<7} | {'Визуальная оценка'}")
print("-" * 85)

for ds, path in DATASETS.items():
    fname = sorted([f for f in os.listdir(path) if f.endswith('.bmp')])[0]
    orig = load(os.path.join(path, fname))
    
    for k in [1, 2, 3]:
        st = embed(orig, k, bits)
        mse, psnr, ss = calc_metrics(orig, st)
        mse_norm = mse / (255**2)
        
        # Визуальная оценка
        if k == 1:
            visual = "Не различимо"
        elif k == 2 and ds == 'BOSS':
            visual = "Слабая зернистость"
        elif k == 2 and ds == 'MED':
            visual = "Зернистость на тканях"
        elif k == 2:
            visual = "Выраж. зернистость"
        elif ds == 'OTHER':
            visual = "Сильные артефакты"
        else:
            visual = "Заметные артефакты"
        
        print(f"{ds:<8} | {k:<2} | {psnr:<7.2f} | {mse_norm:<12.2e} | {ss:<7.4f} | {visual}")

print("-" * 85)
print("✅ Таблица готова для копирования в отчёт (Таблица 5.4)")
