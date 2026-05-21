import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv('results/tables/metrics_all.csv')
    groups = df.groupby(['dataset', 'method'])
    rows = []
    for (dataset, method), g in groups:
        ps = g['PSNR'].dropna().values
        n = len(ps)
        if n == 0: continue
        mean = np.mean(ps)
        se = np.std(ps, ddof=1) / np.sqrt(n)
        tval = stats.t.ppf(0.975, df=n-1)
        low, high = mean - tval*se, mean + tval*se
        rows.append({'dataset': dataset, 'method': method, 'n': n, 'mean_PSNR': mean, 'CI_low': low, 'CI_high': high})

    out = pd.DataFrame(rows)
    os.makedirs('results/tables', exist_ok=True)
    out.to_csv('results/tables/psnr_ci.csv', index=False)
    print('✅ Saved CI to results/tables/psnr_ci.csv')

    plt.figure(figsize=(9,5))
    datasets = out['dataset'].unique()
    for ds in datasets:
        sub = out[out['dataset']==ds]
        offset = {'BOSS': -0.2, 'MED': 0.0, 'OTHER': 0.2}.get(ds, 0)
        x = np.arange(len(sub)) + offset
        plt.errorbar(x, sub['mean_PSNR'], yerr=[sub['mean_PSNR']-sub['CI_low'], sub['CI_high']-sub['mean_PSNR']], fmt='o', label=ds)
    
    plt.xticks([0, 1], ['LSB', 'ADAPT_GRADIENT'])
    plt.ylabel('PSNR (dB)')
    plt.title('Mean PSNR with 95% CI by Dataset and Method')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    
    os.makedirs('results/figures', exist_ok=True) 
    plt.savefig('results/figures/psnr_ci.png', dpi=150)
    print('✅ Saved plot to results/figures/psnr_ci.png')

if __name__ == '__main__':
    main()
