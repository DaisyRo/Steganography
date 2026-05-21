import os, sys
from PIL import Image
from tqdm import tqdm

def convert_image(in_path, out_path):
    try:
        img = Image.open(in_path).convert('L')          # 8-bit grayscale
        img = img.resize((512, 512), Image.Resampling.LANCZOS)  # 512x512
        img.save(out_path, format='BMP')                # Сохраняем как BMP
        return True
    except Exception as e:
        print(f"⚠️ Ошибка {in_path}: {e}")
        return False

def batch_convert(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        print(f"❌ Директория '{src_dir}' не найдена.")
        sys.exit(1)
    os.makedirs(dst_dir, exist_ok=True)
    
    exts = ('.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp')
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    if not files:
        print(f"❌ В '{src_dir}' нет поддерживаемых изображений.")
        return

    print(f"📦 Найдено {len(files)} файлов. Конвертирую...")
    ok = sum(1 for f in tqdm(files, desc="Обработка") 
             if convert_image(os.path.join(src_dir, f), 
                              os.path.join(dst_dir, os.path.splitext(f)[0]+'.bmp')))
    print(f"✅ Готово: {ok}/{len(files)} успешно сохранены в {dst_dir}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Использование: python3 code/convert_to_bmp.py <исходники> <результат>")
        print("Пример: python3 code/convert_to_bmp.py ~/Downloads/boss_raw data/bmp_boss")
        sys.exit(1)
    batch_convert(sys.argv[1], sys.argv[2])
