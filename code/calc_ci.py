import os, struct, numpy as np
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = {'BOSS': 'data/bmp_boss', 'MED': 'data/bmp_med', 'OTHER': 'data/bmp_other'}
SECRET = 'code/secret.bin'
OUT_DIR = 'results/tables'
os.makedirs(OUT_DIR, exist_ok=True)

# Подготовка битов сообщения
with open(SECRET, 'rb') as f: msg = f.read()
header = struct.pack('>I', len(msg))
full_bits = np.array([(b >> i) & 1 for b in (header + msg) for i in range(7, -1, -1)], dtype=np.uint8)
n_bits = len(full_bits)

results = {ds: {k: {'psnr': [], 'var': []} for k in [1,2,3]} for ds in DATASETS}
print("🔄 Расчёт метрик для всех 300 изображений...")

for ds, path in DATASETS.items():
    files = [f for f in os.listdir(path) if f.endswith('.bmp')]
    for fname in files:
        arr = np.array(Image.open(os.path.join(path, fname)).convert('L'), dtype=np.uint8)
        flat = arr.flatten()
        bits_to_use = min(n_bits, flat.size)

        for k in [1, 2, 3]:
            mask = 1 << (k-1)
            inv_mask = 0xFF ^ mask
            
            # Внедрение
            stego = flat.copy()
            stego[:bits_to_use] = (stego[:bits_to_use] & inv_mask) | (full_bits[:bits_to_use] << (k-1))
            stego = stego.reshape(arr.shape)
            
            # PSNR
            mse = np.mean((arr - stego)**2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100.0
            results[ds][k]['psnr'].append(psnr)
            
            # Дисперсия k-й битовой плоскости
            bit_plane = (arr >> (k-1)) & 1
            var = np.var(bit_plane)
            results[ds][k]['var'].append(var)

# Расчёт 95% ДИ
alpha = 0.05
n = 100
t_val = stats.t.ppf(1 - alpha/2, n-1)
chi2_l = stats.chi2.ppf(alpha/2, n-1)
chi2_u = stats.chi2.ppf(1 - alpha/2, n-1)

psnr_rows, var_rows = [], []
for ds in DATASETS:
    for k in [1,2,3]:
        p_arr = np.array(results[ds][k]['psnr'])
        v_arr = np.array(results[ds][k]['var'])
        
        mean_p, std_p = np.mean(p_arr), np.std(p_arr, ddof=1)
        ci_p = t_val * std_p / np.sqrt(n)
        
        mean_v = np.mean(v_arr)
        ci_v_l = (n-1)*mean_v / chi2_u
        ci_v_u = (n-1)*mean_v / chi2_l
        ci_v_half = (ci_v_u - ci_v_l) / 2
        
        psnr_rows.append([ds, k, mean_p, ci_p])
        var_rows.append([ds, k, mean_v, ci_v_half])

# Сохранение таблиц
pd.DataFrame(psnr_rows, columns=['Набор', 'k', 'Mean_PSNR', 'CI_PSNR']).to_csv(f'{OUT_DIR}/psnr_ci.csv', index=False)
pd.DataFrame(var_rows, columns=['Набор', 'k', 'Mean_Var', 'CI_Var']).to_csv(f'{OUT_DIR}/var_ci.csv', index=False)

# Вывод в консоль для отчёта
print("\n📊 ТАБЛИЦА 1. Доверительные интервалы PSNR (дБ), α=0.05")
print(f"{'Набор':<8} | {'k':<2} | {'Среднее':<10} | {'95% ДИ (±)':<10}")
print("-"*40)
for r in psnr_rows: print(f"{r[0]:<8} | {r[1]:<2} | {r[2]:<10.3f} | {r[3]:<10.3f}")

print("\n ТАБЛИЦА 2. Доверительные интервалы дисперсии битовых плоскостей")
print(f"{'Набор':<8} | {'k':<2} | {'Среднее':<10} | {'95% ДИ (±)':<10}")
print("-"*40)
for r in var_rows: print(f"{r[0]:<8} | {r[1]:<2} | {r[2]:<10.4f} | {r[3]:<10.4f}")

# Графики
plt.figure(figsize=(8,5))
for ds in DATASETS:
    means = [results[ds][k]['psnr'] for k in [1,2,3]]
    cis = [t_val * np.std(m, ddof=1)/np.sqrt(n) for m in means]
    plt.errorbar([1,2,3], [np.mean(m) for m in means], yerr=cis, fmt='o-', label=ds, capsize=5)
plt.xticks([1,2,3])
plt.xlabel('Номер битовой плоскости (k)')
plt.ylabel('PSNR (дБ)')
plt.title('Среднее PSNR и 95% доверительные интервалы по наборам')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/psnr_ci_plot.png', dpi=150)
print("\n✅ Графики и таблицы сохранены в results/tables/")
