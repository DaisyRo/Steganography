import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11

METRICS_CSV = 'results/tables/metrics_all.csv'
OUT_DIR_TABLES = 'results/tables'
OUT_DIR_FIGS = 'results/figures'
STEGO_DIR = 'results/stego_examples'
DATASET_PATHS = {
    'BOSS': 'data/bmp_boss',
    'MED':  'data/bmp_med',
    'OTHER': 'data/bmp_other'
}

os.makedirs(OUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

def ci_mean(data, alpha=0.05):
    """Доверительный интервал для среднего (t-распределение)"""
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    mean = np.mean(data)
    se = stats.sem(data)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return (mean, lo, hi, n)

def ci_variance(data, alpha=0.05):
    """Доверительный интервал для дисперсии (хи-квадрат)"""
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    s2 = np.var(data, ddof=1)
    chi_lo = stats.chi2.ppf(alpha/2, df=n-1)
    chi_hi = stats.chi2.ppf(1-alpha/2, df=n-1)
    lower = (n-1) * s2 / chi_hi
    upper = (n-1) * s2 / chi_lo
    return (s2, lower, upper, n)

def compute_psnr_summary(df_metrics):
    rows = []
    for ds in sorted(df_metrics['dataset'].unique()):
        for k in sorted(df_metrics['k'].unique()):
            vals = df_metrics[(df_metrics.dataset==ds)&(df_metrics.k==k)]['PSNR'].dropna().values
            mean, lo, hi, n = ci_mean(vals)
            rows.append({'dataset': ds, 'k': k, 'PSNR_mean': mean, 'PSNR_lo': lo, 'PSNR_hi': hi, 'n': n})
    return pd.DataFrame(rows)

def plot_psnr_ci(psnr_ci_df, outpath):
    plt.figure(figsize=(9, 5))
    sns.pointplot(data=psnr_ci_df, x='k', y='PSNR_mean', hue='dataset',
                  dodge=0.25, join=False, capsize=0.12, errwidth=1.2)
    # Рисуем интервалы вручную для точности
    for _, row in psnr_ci_df.iterrows():
        offset = 0.0
        if row['dataset'] == 'MED': offset = 0.15
        elif row['dataset'] == 'OTHER': offset = -0.15
        x = row['k'] + offset
        plt.plot([x, x], [row['PSNR_lo'], row['PSNR_hi']], color='gray', linewidth=1.2)
        plt.plot([x-0.04, x+0.04], [row['PSNR_lo'], row['PSNR_lo']], color='gray', linewidth=1)
        plt.plot([x-0.04, x+0.04], [row['PSNR_hi'], row['PSNR_hi']], color='gray', linewidth=1)
    plt.title('Mean PSNR with 95% Confidence Intervals')
    plt.xlabel('Bit plane k')
    plt.ylabel('PSNR (dB)')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"✓ График сохранён: {outpath}")

def plot_psnr_box(df_metrics, outpath):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_metrics, x='k', y='PSNR', hue='dataset', showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black"})
    plt.title('PSNR Distribution by Dataset and Bit Plane')
    plt.xlabel('Bit plane k')
    plt.ylabel('PSNR (dB)')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"✓ График сохранён: {outpath}")

def plot_histograms_sample(ds_map, outdir):
    for ds, path in ds_map.items():
        if not os.path.isdir(path): continue
        files = sorted([f for f in os.listdir(path) if f.lower().endswith('.bmp')])
        if not files: continue
        fname = files[0]
        img_path = os.path.join(path, fname)
        try:
            orig = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
        except: continue
        stego_name = f'{ds}_{os.path.splitext(fname)[0]}_k1.bmp'
        stego_path = os.path.join(STEGO_DIR, stego_name)
        if os.path.exists(stego_path):
            try:
                stego = np.array(Image.open(stego_path).convert('L'), dtype=np.uint8)
            except: stego = orig
        else:
            stego = orig
        plt.figure(figsize=(6, 4))
        plt.hist(orig.flatten(), bins=64, alpha=0.6, label='Original', color='tab:blue', edgecolor='black')
        plt.hist(stego.flatten(), bins=64, alpha=0.6, label='Stego (k=1)', color='tab:orange', edgecolor='black')
        plt.title(f'Histogram: {ds} — {fname}')
        plt.xlabel('Intensity (0–255)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        outp = os.path.join(outdir, f'{ds}_{os.path.splitext(fname)[0]}_hist.png')
        plt.savefig(outp, dpi=200)
        plt.close()
        print(f"✓ Гистограмма: {outp}")

def main():
    print("📊 Чтение метрик...")
    if not os.path.exists(METRICS_CSV):
        print(f"❌ Не найден файл: {METRICS_CSV}")
        print("Сначала запустите batch_process.py")
        return
    df_metrics = pd.read_csv(METRICS_CSV)
    
    # 1. CI для PSNR
    print("📈 Расчёт доверительных интервалов для PSNR...")
    psnr_ci = compute_psnr_summary(df_metrics)
    psnr_ci_path = os.path.join(OUT_DIR_TABLES, 'psnr_ci.csv')
    psnr_ci.to_csv(psnr_ci_path, index=False)
    print(f"✓ Сохранено: {psnr_ci_path}")
    
    # 2. Графики PSNR
    print(" Построение графиков...")
    plot_psnr_ci(psnr_ci, os.path.join(OUT_DIR_FIGS, 'psnr_mean_ci.png'))
    plot_psnr_box(df_metrics, os.path.join(OUT_DIR_FIGS, 'psnr_boxplot.png'))
    
    # 3. Гистограммы (примеры)
    print(" Построение гистограмм...")
    try:
        from PIL import Image
        plot_histograms_sample(DATASET_PATHS, OUT_DIR_FIGS)
    except ImportError:
        print("⚠️  PIL не установлен, пропускаем гистограммы")
    
    # 4. Вывод сводки
    print("\n Сводка по доверительным интервалам PSNR:")
    print(psnr_ci.to_string(index=False))
    
    print(f"\n Готово! Результаты в:")
    print(f"  • Таблицы: {OUT_DIR_TABLES}/")
    print(f"  • Графики: {OUT_DIR_FIGS}/")

if __name__ == '__main__':
    main()
