import os
import numpy as np
from PIL import Image
import math

def entropy(bits):
    """Расчет бинарной энтропии (0.0 - полная структура, 1.0 - полный шум)"""
    p1 = np.mean(bits)
    if p1 == 0 or p1 == 1:
        return 0.0
    return -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))

def analyze_image(img_path):
    """Анализ битовых плоскостей одного изображения"""
    arr = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
    results = {}
    for k in range(1, 9):
        # Извлекаем k-й бит: сдвиг вправо, маска 1
        bit_plane = (arr >> (k - 1)) & 1
        results[k] = entropy(bit_plane)
    return results

def main():
    datasets = {
        'BOSS': 'data/bmp_boss',
        'MED': 'data/bmp_med',
        'OTHER': 'data/bmp_other'
    }

    print(f"{'Набор':<10} | {'k=1':<5} | {'k=2':<5} | {'k=3':<5} | {'k=4':<5} | {'k=5':<5} | {'k=6':<5}")
    print("-" * 65)

    for name, path in datasets.items():
        files = [f for f in os.listdir(path) if f.endswith('.bmp')]
        if not files: continue
        
        # Берем первое изображение для примера
        img_path = os.path.join(path, files[0])
        ent = analyze_image(img_path)
        
        # Вывод только для k=1..6 (как просит задание)
        row = [f"{ent[k]:.2f}" for k in range(1, 7)]
        print(f"{name:<10} | {' | '.join(row)}")

    print("-" * 65)
    print("Интерпретация: Чем ближе к 1.00, тем больше плоскость похожа на случайный шум.")
    print("Значения < 0.80 указывают на наличие структур (контуров, однотонных областей).")

if __name__ == '__main__':
    main()
