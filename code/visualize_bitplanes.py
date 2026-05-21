import os
import numpy as np
from PIL import Image

# Пути к наборам данных
DATASETS = {
    'BOSS': 'data/bmp_boss',
    'MED': 'data/bmp_med',
    'OTHER': 'data/bmp_other'
}

# Папка для результатов исследования
OUT_DIR = 'results/research'
os.makedirs(OUT_DIR, exist_ok=True)

def extract_plane(arr, k):
    """
    Извлекает k-ю битовую плоскость.
    k=1 - младший бит, k=8 - старший бит.
    Возвращает массив 0 и 255 (черный/белый).
    """
    # Сдвигаем бит вправо, оставляем последний, умножаем на 255
    return ((arr >> (k - 1)) & 1) * 255

def process_dataset(ds_name, ds_path, out_dir):
    """
    Обрабатывает 5 изображений из набора ds_path.
    Создает сетку 2x4 с битовыми плоскостями k=1..8.
    """
    if not os.path.exists(ds_path):
        print(f"⚠️ Папка {ds_path} не найдена, пропускаем.")
        return

    # Получаем список BMP файлов и берем первые 5
    files = sorted([f for f in os.listdir(ds_path) if f.endswith('.bmp')])[:5]
    
    print(f"📊 {ds_name}: обработка {len(files)} изображений...")

    for filename in files:
        img_path = os.path.join(ds_path, filename)
        out_filename = f"{ds_name}_{os.path.splitext(filename)[0]}_bitplanes.png"
        out_path = os.path.join(out_dir, out_filename)

        # Загрузка изображения
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape

        # Создаем холст для сетки 2x4 (ширина 4, высота 2)
        grid_h, grid_w = h * 2, w * 4
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        # Заполняем сетку плоскостями k=1..8
        for k in range(1, 9):
            plane = extract_plane(arr, k)
            
            # Определяем координаты (столбец 0-3, строка 0-1)
            col = (k - 1) % 4
            row = (k - 1) // 4
            
            y_start = row * h
            x_start = col * w
            
            # Копируем плоскость на холст
            grid[y_start:y_start+h, x_start:x_start+w] = plane

        # Сохраняем результат
        Image.fromarray(grid).save(out_path)
        print(f"   ✅ Сохранено: {out_path}")

def main():
    print(" Запуск массовой визуализации битовых плоскостей")
    print("-" * 50)
    
    for name, path in DATASETS.items():
        process_dataset(name, path, OUT_DIR)
        
    print("-" * 50)
    print(f"✅ Готово! Результаты в папке: {OUT_DIR}")

if __name__ == '__main__':
    main()
