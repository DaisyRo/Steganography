import argparse
import os
import sys
import struct
import numpy as np
from PIL import Image

def load_image(path):
    """Загрузка 8-bit grayscale изображения"""
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)

def save_image(arr, path):
    """Сохранение как 8-bit BMP"""
    Image.fromarray(arr.astype(np.uint8)).save(path, format='BMP')

def bytes_to_bits(data):
    """Преобразование байтов в список бит (MSB first)"""
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_bytes(bits):
    """Преобразование списка бит в байты"""
    out = bytearray()
    cur = 0
    for i, b in enumerate(bits):
        cur = (cur << 1) | b
        if (i + 1) % 8 == 0:
            out.append(cur)
            cur = 0
    return bytes(out)

def mode_extract_plane(args):
    """Режим 1: Извлечение битовой плоскости"""
    arr = load_image(args.image)
    k = args.k
    bit = (arr >> (k - 1)) & 1
    out_arr = (bit * 255).astype(np.uint8)
    save_image(out_arr, args.output)
    print(f"✓ Битовая плоскость k={k} сохранена в {args.output}")

def mode_embed(args):
    """Режим 2: Внедрение сообщения"""
    if not os.path.exists(args.message):
        print(f"❌ Файл сообщения не найден: {args.message}")
        sys.exit(1)
    with open(args.message, 'rb') as f:
        msg = f.read()
    if len(msg) < 30 * 1024:
        print(f"⚠️  Предупреждение: размер сообщения {len(msg)} байт < 30 КБ (по ТЗ)")

    arr = load_image(args.image)
    H, W = arr.shape
    capacity = H * W

    # Заголовок: 4 байта (32 бита) с длиной сообщения для точного извлечения
    header = struct.pack('>I', len(msg))
    full_data = header + msg
    bits = bytes_to_bits(full_data)
    bits_to_write = min(len(bits), capacity)

    flat = arr.flatten().copy()
    mask = 1 << (args.k - 1)
    inv_mask = 0xFF ^ mask

    for i in range(bits_to_write):
        if bits[i] == 1:
            flat[i] |= mask
        else:
            flat[i] &= inv_mask

    out_arr = flat.reshape(H, W)
    save_image(out_arr, args.output)
    print(f"✓ Внедрено {bits_to_write} бит ({bits_to_write // 8} байт) в {args.output}")
    print(f"  Плоскость: k={args.k}, Ёмкость контейнера: {capacity} бит")

def mode_extract_msg(args):
    """Режим 3: Извлечение сообщения"""
    arr = load_image(args.image)
    flat = arr.flatten()
    mask = 1 << (args.k - 1)

    if len(flat) < 32:
        print("❌ Изображение слишком маленькое для заголовка длины")
        sys.exit(1)

    # Читаем заголовок (первые 32 бита)
    header_bits = [(flat[i] & mask) >> (args.k - 1) for i in range(32)]
    header_bytes = bits_to_bytes(header_bits)
    msg_len = struct.unpack('>I', header_bytes)[0]

    total_bits = 32 + msg_len * 8
    if total_bits > len(flat):
        print(f"⚠️  Ёмкость меньше заявленной длины. Извлекаем {len(flat)//8} байт.")
        total_bits = len(flat)

    all_bits = [(flat[i] & mask) >> (args.k - 1) for i in range(total_bits)]
    msg_bytes = bits_to_bytes(all_bits[32:])

    with open(args.output, 'wb') as f:
        f.write(msg_bytes)
    print(f"✓ Извлечено {len(msg_bytes)} байт в {args.output}")

def main():
    parser = argparse.ArgumentParser(description="LSB Steganography Tool (Задание 1)")
    parser.add_argument('--mode', choices=['extract-plane', 'embed', 'extract-msg'], required=True,
                        help="Режим работы")
    parser.add_argument('--image', required=True, help="Путь к BMP изображению")
    parser.add_argument('--k', type=int, choices=range(1, 9), required=True,
                        help="Номер битовой плоскости (1..8)")
    parser.add_argument('--message', help="Путь к файлу сообщения (только для embed)")
    parser.add_argument('--output', required=True, help="Путь для сохранения результата")
    args = parser.parse_args()

    if args.mode == 'extract-plane':
        mode_extract_plane(args)
    elif args.mode == 'embed':
        if not args.message:
            print("❌ Для режима embed укажите --message")
            sys.exit(1)
        mode_embed(args)
    elif args.mode == 'extract-msg':
        mode_extract_msg(args)

if __name__ == '__main__':
    main()
