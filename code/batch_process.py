import os, argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from bitplane_tool import load_image, save_image, bits_from_bytes, pack_payload, unpack_payload
from embed_lsb import embed_lsb, extract_lsb
from embed_adaptive import embed_adaptive
from evaluate import compute_metrics

DATASETS = {
    'BOSS': 'data/bmp_boss',
    'MED': 'data/bmp_med',
    'OTHER': 'data/bmp_other'
}

def main(args):
    out_dir = args.out
    os.makedirs(f'{out_dir}/stego_examples', exist_ok=True)
    os.makedirs(f'{out_dir}/tables', exist_ok=True)

    with open(args.logo, 'rb') as f:
        logo_bytes = f.read()
    payload = pack_payload(logo_bytes)
    bits = bits_from_bytes(payload)

    rows = []
    for ds_name, ds_path in DATASETS.items():
        files = sorted([f for f in os.listdir(ds_path) if f.lower().endswith(('.bmp', '.png'))])
        for fname in tqdm(files, desc=f'Processing {ds_name}'):
            img_path = os.path.join(ds_path, fname)
            orig = load_image(img_path)

            # LSB
            stego_lsb, n_w_lsb, sel_lsb = embed_lsb(orig.copy(), bits, k=args.k, key=args.key, capacity_fraction=args.capacity)
            metrics_lsb = compute_metrics(orig, stego_lsb)
            rows.append({'dataset': ds_name, 'image': fname, 'method': 'LSB', 'n_written': n_w_lsb, **metrics_lsb})
            save_image(stego_lsb, f'{out_dir}/stego_examples/{ds_name}_{os.path.splitext(fname)[0]}_lsb.png')

            # Adaptive
            stego_ad, n_w_ad, sel_ad = embed_adaptive(orig.copy(), bits, k=args.k, capacity_fraction=args.capacity, key=args.key)
            metrics_ad = compute_metrics(orig, stego_ad)
            rows.append({'dataset': ds_name, 'image': fname, 'method': 'ADAPT_GRADIENT', 'n_written': n_w_ad, **metrics_ad})
            save_image(stego_ad, f'{out_dir}/stego_examples/{ds_name}_{os.path.splitext(fname)[0]}_adapt.png')

    df = pd.DataFrame(rows)
    csv_path = f'{out_dir}/tables/metrics_all.csv'
    df.to_csv(csv_path, index=False)
    print(f'✅ Saved metrics to {csv_path}')

    # Быстрая проверка извлечения на первом файле BOSS
    test_path = f'{out_dir}/stego_examples/BOSS_{os.path.splitext(files[0])[0]}_lsb.png'
    stego_test = load_image(test_path)
    ext_bits = extract_lsb(stego_test, k=args.k, key=args.key, n_bits=len(bits))
    ext_bytes = bytes(ext_bits[:len(payload)*8])
    ok, data = unpack_payload(ext_bytes)
    print(f'🔍 Extraction test: {"✅ SUCCESS" if ok else "❌ FAILED"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    parser.add_argument('--logo', default='code/logo.png')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--key', default='secret')
    parser.add_argument('--capacity', type=float, default=0.5)
    main(parser.parse_args())
