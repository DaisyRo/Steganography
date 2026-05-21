import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def load_image(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)

def save_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, format='BMP')

def get_bits_from_bytes(data):
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def embed_bitstream(img_arr, k, msg_bits):
    flat = img_arr.flatten().copy()
    mask = 1 << (k - 1)
    inv_mask = 0xFF ^ mask
    count = 0
    for i in range(min(len(msg_bits), len(flat))):
        if msg_bits[i] == 1:
            flat[i] |= mask
        else:
            flat[i] &= inv_mask
        count += 1
    return flat.reshape(img_arr.shape), count

def calc_mse(orig, stego):
    return np.mean((orig - stego) ** 2)

def calc_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def calc_ssim(orig, stego):
    return ssim(orig, stego, data_range=255)

def main():
    datasets = {
        'BOSS': 'data/bmp_boss',
        'MED': 'data/bmp_med',
        'OTHER': 'data/bmp_other'
    }
    secret_path = 'code/secret.bin'
    if not os.path.exists(secret_path):
        print("❌ Ошибка: code/secret.bin не найден. Сначала создайте его.")
        return

    with open(secret_path, 'rb') as f:
        secret_data = f.read()

    msg_bits = get_bits_from_bytes(secret_data)

    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/stego_examples', exist_ok=True)

    results = []
    print("🚀 Запуск пакетной обработки (внедрение k=1,2,3)...")

    for ds_name, ds_path in datasets.items():
        if not os.path.exists(ds_path):
            print(f"⚠️ Пропускаем {ds_path} (не найден)")
            continue

        files = sorted([f for f in os.listdir(ds_path) if f.lower().endswith('.bmp')])
        for fname in tqdm(files, desc=f"Обработка {ds_name}"):
            img_path = os.path.join(ds_path, fname)
            try:
                orig = load_image(img_path)
                for k in (1, 2, 3):
                    stego, written = embed_bitstream(orig, k, msg_bits)
                    mse = calc_mse(orig, stego)
                    psnr = calc_psnr(mse)
                    ssim_val = calc_ssim(orig, stego)

                    results.append({
                        'dataset': ds_name,
                        'image': fname,
                        'k': k,
                        'n_bits_written': written,
                        'MSE': round(mse, 4),
                        'PSNR': round(psnr, 4),
                        'SSIM': round(ssim_val, 4)
                    })

                    # Сохраняем примеры для первых файлов
                    if ds_name == 'BOSS' and fname.startswith(('1', '2', '3')):
                        out_stego = f"results/stego_examples/{ds_name}_{os.path.splitext(fname)[0]}_k{k}.bmp"
                        save_image(stego, out_stego)
            except Exception as e:
                print(f"❌ Ошибка в {fname}: {e}")

    if results:
        df = pd.DataFrame(results)
        csv_path = 'results/tables/metrics_all.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Готово! Метрики сохранены в: {csv_path}")
        print(f" Всего записей: {len(results)}")
    else:
        print("❌ Не обработано ни одного файла.")

if __name__ == '__main__':
    main()
