from PIL import Image, ImageDraw, ImageFont
import os

def add_labels_to_bitplanes(input_path, output_path):
    """Добавляет рамку и подписи рядов к изображению битовых плоскостей"""
    
    # Открываем изображение
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    
    # Добавляем отступы: сверху 40px, снизу 30px, по бокам 10px
    new_w, new_h = w + 20, h + 70
    new_img = Image.new('RGB', (new_w, new_h), color='white')
    new_img.paste(img, (10, 40))
    
    draw = ImageDraw.Draw(new_img)
    
    # Попытка загрузить шрифт, если нет — используем дефолтный
    try:
        # Попробуем найти стандартный шрифт Linux
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=14)
            font_small = ImageFont.truetype(font_path, size=11)
        else:
            font = ImageFont.load_default()
            font_small = font
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Рисуем рамку вокруг самого изображения битовых плоскостей
    draw.rectangle([10, 40, 10+w, 40+h], outline='black', width=2)
    
    # Подписи рядов (по центру)
    draw.text((new_w//2, 15), 'Верхний ряд: k=1 (LSB) → k=4', 
              fill='black', font=font, anchor='mt')
    draw.text((new_w//2, new_h-15), 'Нижний ряд: k=5 → k=8 (MSB)', 
              fill='black', font=font, anchor='mb')
    
    # Подписи отдельных плоскостей (мелким шрифтом, под рамкой)
    labels = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8']
    cell_w = w // 4
    for i, label in enumerate(labels):
        x = 10 + (i % 4) * cell_w + cell_w // 2
        y = 40 + h + 12 if i >= 4 else 40 - 8  # над верхним рядом или под нижним
        draw.text((x, y), label, fill='gray', font=font_small, anchor='mm')
    
    # Сохраняем результат
    new_img.save(output_path, quality=95)
    print(f"✓ {output_path}")

if __name__ == '__main__':
    # Обрабатываем все файлы битовых плоскостей в папке research
    research_dir = 'results/research'
    for fname in os.listdir(research_dir):
        if fname.endswith('_bitplanes.png') and '_labeled' not in fname:
            in_path = os.path.join(research_dir, fname)
            out_path = os.path.join(research_dir, fname.replace('.png', '_labeled.png'))
            add_labels_to_bitplanes(in_path, out_path)
    print("✅ Готово! Файлы с подписями сохранены с суффиксом _labeled")
