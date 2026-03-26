#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# генератор всего датасета - буквы цифры пунктуация слова предложения
# ОПТИМИЗИРОВАНО ДЛЯ MACBOOK AIR M1: 16 процессов, spawn context, быстрые операции
# Запуск: python3 generate.py
#   Data-set/Train/Letters/   - отдельные буквы
#   Data-set/Train/Nums/      - цифры
#   Data-set/Train/Punkt/     - знаки пунктуации
#   Data-set/Train/Words/     - слова
#   Data-set/Train/Sentences/ - абзацы и предложения

import os
import cv2
import sys
import time
import random
import platform
import multiprocessing as mp
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


# настройки - менять только здесь 
class Config:
    BASE_DIR     = "Data-set/Train"   # корневая папка

    LETTERS_DIR  = "Data-set/Train/Letters"
    NUMS_DIR     = "Data-set/Train/Nums"
    PUNKT_DIR    = "Data-set/Train/Punkt"
    WORDS_DIR    = "Data-set/Train/Words"
    SENTENCES_DIR = "Data-set/Train/Sentences"

    FONTS_DIR    = "fonts"            # папка со шрифтами
    WORDS_FILE   = "russian_words.txt" # словарь - скачай заранее

    LABELS_FILE  = "labels.txt"       # имя файла меток в каждой папке

    # сколько генерировать
    LETTERS_PER_CHAR  = 50
    NUMS_PER_CHAR     = 50
    PUNKT_PER_CHAR    = 50
    WORDS_COUNT       = 1000
    SENTENCES_COUNT   = 500

    # размеры изображений
    CHAR_H       = 64     # высота для букв цифр пунктуации
    CHAR_W       = 64     # ширина
    WORD_H       = 64     # высота для слов
    SENTENCE_H   = 128    # высота для предложений (больше текста)
    SENTENCE_W   = 1024   # ширина для предложений

    # параллельность
    if platform.system() == "Darwin":  # macOS M1 оптимизация
        NUM_WORKERS  = mp.cpu_count() * 2  # 16 процессов для 8 ядер
    else:
        NUM_WORKERS  = mp.cpu_count() or 4
    LABEL_FLUSH  = 500    # буфер записи меток

    # символы которые генерируем
    RUS_LOWER  = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    RUS_UPPER  = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    DIGITS     = "0123456789"
    PUNKT      = ".,!?:;-—()[]«»\"'"

    # черный список шрифтов
    FONT_BLACKLIST = {
        "Freestyle_Script_Bold__RUS",
        "KosolapaScript_Regular",
        "letov",
    }

    # минимальные параметры предложений
    MIN_WORDS_IN_SENTENCE = 5
    MAX_WORDS_IN_SENTENCE = 20
    MIN_SENTENCES_IN_PARA = 2
    MAX_SENTENCES_IN_PARA = 5


C = Config()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# безопасные имена для файлов пунктуации
SAFE_NAMES = {
    ".": "point", ",": "comma", "!": "excl", "?": "quest",
    ":": "colon", ";": "semicol", "-": "dash", "—": "emdash",
    "(": "paren_o", ")": "paren_c", "[": "brack_o", "]": "brack_c",
    "«": "guilm_o", "»": "guilm_c", '"': "dquote", "'": "squote",
}


# сканирование шрифтов
def scan_fonts() -> list:
    FONT_EXTS = {".ttf", ".otf", ".ttc"}
    os.makedirs(C.FONTS_DIR, exist_ok=True)
    fonts = []

    for fn in sorted(os.listdir(C.FONTS_DIR)):
        if Path(fn).suffix.lower() not in FONT_EXTS:
            continue
        stem = Path(fn).stem
        if stem in C.FONT_BLACKLIST:
            print(f"  skip  {fn}")
            continue
        path = os.path.join(C.FONTS_DIR, fn)
        if os.path.getsize(path) > 1000:
            fonts.append(path)

    if not fonts:
        sdirs = []
        if platform.system() == "Windows":
            sdirs = [r"C:\Windows\Fonts"]
        elif platform.system() == "Darwin":
            sdirs = ["/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
        else:
            sdirs = ["/usr/share/fonts", os.path.expanduser("~/.fonts")]

        print("  шрифтов в fonts/ нет - ищем системные...")
        for sd in sdirs:
            if not os.path.isdir(sd):
                continue
            for root, _, files in os.walk(sd):
                for fn in files:
                    if Path(fn).suffix.lower() in FONT_EXTS:
                        if Path(fn).stem not in C.FONT_BLACKLIST:
                            fonts.append(os.path.join(root, fn))

    if not fonts:
        print("  ошибка: нет шрифтов - положи .ttf в папку fonts/")
        sys.exit(1)

    print(f"  шрифтов: {len(fonts)}")
    return fonts


# загрузка словаря
def load_words() -> list:
    if not Path(C.WORDS_FILE).exists():
        print(f"  файл не найден: {C.WORDS_FILE}")
        print("  скачай: https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt")
        return []

    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            with open(C.WORDS_FILE, "r", encoding=enc) as f:
                words = [ln.strip() for ln in f if ln.strip()]
            print(f"  словарь: {len(words):,} слов ({enc})")
            return words
        except UnicodeDecodeError:
            continue
    return []


# рендер текста - возвращает numpy array (оптимизировано для M1)
def render_text(text: str, font_path: str, img_h: int, img_w: int = None) -> np.ndarray:
    try:
        font = None
        for fs in range(52, 10, -2):
            f    = ImageFont.truetype(font_path, fs)
            bbox = f.getbbox(text)
            h    = bbox[3] - bbox[1]
            w    = bbox[2] - bbox[0]
            fits_h = h <= img_h * 0.85
            fits_w = (img_w is None) or (w <= img_w * 0.95)
            if fits_h and fits_w:
                font = f
                break
        if font is None:
            font = ImageFont.truetype(font_path, 10)

        bbox = font.getbbox(text)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        pad  = 20
        cw   = img_w if img_w else max(tw + pad * 2, 60)
        ch   = img_h + 16

        # Используем RGBA для лучшей производительности на M1
        img  = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        x = (cw - tw) // 2 - bbox[0] + random.randint(-3, 3)
        y = (ch - th) // 2 - bbox[1] + random.randint(-3, 3)
        draw.text((x, y), text, font=font, fill=(random.randint(10, 60), 0, 0, 255))

        # Конвертируем в grayscale быстрее
        img = img.convert("L")
        img   = ImageOps.crop(img, (0, 4, 0, 4))
        scale = img_h / img.height
        nw    = img_w if img_w else max(int(img.width * scale), 16)
        img   = img.resize((nw, img_h), Image.Resampling.LANCZOS)

        return np.array(img, dtype=np.uint8)

    except Exception:
        return None


# фон
def make_background(w: int, h: int) -> np.ndarray:
    bg_type = random.choices(
        ["white", "cream", "lined", "grid", "aged"],
        weights=[30, 25, 20, 15, 10]
    )[0]

    if bg_type == "white":
        bg = np.full((h, w), 255, np.uint8)
    elif bg_type == "cream":
        base  = random.randint(232, 250)
        bg    = np.full((h, w), base, np.uint8)
        noise = np.random.randint(-6, 6, (h, w), dtype=np.int16)
        bg    = np.clip(bg.astype(np.int16) + noise, 200, 255).astype(np.uint8)
    elif bg_type == "lined":
        bg   = np.full((h, w), 255, np.uint8)
        step = random.choice([8, 10, 12, 16])
        for y in range(0, h, step):
            bg[y, :] = random.randint(185, 215)
    elif bg_type == "grid":
        bg   = np.full((h, w), 255, np.uint8)
        step = random.choice([8, 10, 12])
        for y in range(0, h, step):
            bg[y, :] = random.randint(190, 215)
        for x in range(0, w, step):
            bg[:, x] = random.randint(190, 215)
    else:  # aged
        base  = random.randint(210, 238)
        bg    = np.full((h, w), base, np.uint8)
        noise = np.random.randint(-18, 18, (h, w), dtype=np.int16)
        bg    = np.clip(bg.astype(np.int16) + noise, 180, 255).astype(np.uint8)
        for _ in range(random.randint(0, 4)):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            cv2.circle(bg, (cx, cy), random.randint(3, 15), random.randint(175, 210), -1)

    return bg


# аугментации
def augment(img: np.ndarray) -> np.ndarray:
    r = random.random

    if r() > 0.35:
        k   = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), random.uniform(0.3, 1.2))

    if r() > 0.25:
        noise = np.random.normal(0, random.uniform(4, 14), img.shape)
        img   = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    if r() > 0.5:
        img = cv2.convertScaleAbs(img,
                                  alpha=random.uniform(0.82, 1.18),
                                  beta=random.randint(-12, 12))

    if r() > 0.55:
        h, w = img.shape
        sf   = random.uniform(-0.06, 0.06)
        M    = np.float32([[1, sf, 0], [0, 1, 0]])
        img  = cv2.warpAffine(img, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_LINEAR)

    if r() > 0.72:
        h, w    = img.shape
        freq    = random.uniform(18, 55)
        amp     = random.uniform(1.0, 2.5)
        phase   = random.uniform(0, 6.28)
        offsets = (amp * np.sin(np.arange(w, dtype=np.float32) / freq + phase)).astype(int)
        map_x   = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        map_y   = (np.tile(np.arange(h), (w, 1)).T + offsets).astype(np.float32)
        img     = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

    if r() > 0.82:
        h, w  = img.shape
        seg_w = max(w // random.randint(3, 7), 1)
        res   = img.copy()
        for i in range(0, w, seg_w):
            off = random.randint(-1, 1)
            res[:, i:i + seg_w] = np.roll(img[:, i:i + seg_w], off, axis=0)
        img = res

    if r() > 0.88:
        h, w = img.shape
        for _ in range(random.randint(1, 3)):
            cv2.circle(img,
                       (random.randint(0, w-1), random.randint(0, h-1)),
                       random.randint(1, 3),
                       int(random.uniform(175, 220)), -1)

    if r() > 0.92:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


# пайплайн одного изображения
def generate_one(text: str, font_path: str, img_h: int, img_w: int = None) -> np.ndarray:
    arr = render_text(text, font_path, img_h, img_w)
    if arr is None:
        return None
    arr = augment(arr)
    bg  = make_background(arr.shape[1], arr.shape[0])
    mask    = arr < 200
    out     = bg.copy()
    out[mask] = arr[mask]
    if random.random() > 0.6:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# сохранение с поддержкой кириллических путей (оптимизировано для M1)
def save_image(path: str, img: np.ndarray):
    # Используем более быстрый метод для M1
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # без сжатия для скорости


# генерация предложений из слов
def build_sentence(words: list) -> str:
    n      = random.randint(C.MIN_WORDS_IN_SENTENCE, C.MAX_WORDS_IN_SENTENCE)
    chosen = random.choices(words, k=n)

    # капитализируем первое слово
    chosen[0] = chosen[0].capitalize()

    sentence = " ".join(chosen)

    # добавляем знак в конце
    end = random.choice([".", ".", ".", "!", "?"])
    sentence += end

    return sentence


def build_paragraph(words: list) -> str:
    n_sentences = random.randint(C.MIN_SENTENCES_IN_PARA, C.MAX_SENTENCES_IN_PARA)
    sentences   = [build_sentence(words) for _ in range(n_sentences)]
    return " ".join(sentences)


# воркер
_fonts_global = []
_words_global = []

def _worker_init(fonts, words):
    global _fonts_global, _words_global
    _fonts_global = fonts
    _words_global = words
    random.seed(os.getpid())
    np.random.seed(os.getpid() % (2 ** 31))


def _worker_fn(task):
    # task = (idx, text, out_dir, img_h, img_w, fname_prefix)
    idx, text, out_dir, img_h, img_w, fname_prefix = task
    font_path = random.choice(_fonts_global)
    img       = generate_one(text, font_path, img_h, img_w)
    if img is None:
        return None
    fname = f"{fname_prefix}_{idx:07d}.png"
    fpath = os.path.join(out_dir, fname)
    save_image(fpath, img)
    del img
    return fname, text


# основная функция генерации группы
def generate_group(tasks_list: list, out_dir: str, group_name: str, fonts: list, words: list):
    os.makedirs(out_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, C.LABELS_FILE)

    # проверяем уже сгенерированное
    done = 0
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        if done > 0:
            print(f"  уже есть {done} - продолжаем с {done}")

    tasks = tasks_list[done:]
    if not tasks:
        print(f"  [{group_name}] всё уже сгенерировано")
        return

    n_workers = C.NUM_WORKERS or max(1, mp.cpu_count())
    success   = 0
    errors    = 0
    label_buf = []
    t_start   = time.perf_counter()

    labels_file = open(labels_path, "a", encoding="utf-8", buffering=1)

    ctx  = mp.get_context("spawn")  # spawn лучше для macOS
    pool = ctx.Pool(
        processes        = n_workers,
        initializer      = _worker_init,
        initargs         = (fonts, words),
        maxtasksperchild = 5000,  # больше задач на процесс для M1
    )

    try:
        with tqdm(total=len(tasks), unit="img", ncols=80,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]"
                  ) as pbar:
            for result in pool.imap_unordered(_worker_fn, tasks,
                                              chunksize=max(1, n_workers * 4)):
                if result is None:
                    errors += 1
                    pbar.update(1)
                    continue
                fname, label = result
                success += 1
                label_buf.append(f"{fname}|{label}\n")
                if len(label_buf) >= C.LABEL_FLUSH:
                    labels_file.writelines(label_buf)
                    label_buf.clear()
                elapsed = time.perf_counter() - t_start
                pbar.set_postfix({
                    "ok":    success,
                    "err":   errors,
                    "img/s": f"{success/elapsed:.0f}" if elapsed > 0 else "0",
                }, refresh=False)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n  прервано - прогресс сохранён")
    finally:
        pool.terminate()
        pool.join()
        if label_buf:
            labels_file.writelines(label_buf)
        labels_file.close()

    elapsed = time.perf_counter() - t_start
    print(f"  [{group_name}] {success} изображений за {elapsed/60:.1f} мин ({success/elapsed:.0f} img/s)")
    if errors:
        print(f"  ошибок: {errors}")


def main():
    print()
    print("+============================================================+")
    print("|        g-vision - генератор всего датасета                |")
    print("+============================================================+")
    print()

    # создаем все папки
    for d in [C.LETTERS_DIR, C.NUMS_DIR, C.PUNKT_DIR, C.WORDS_DIR, C.SENTENCES_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"  папка: {d}")
    print()

    # сканируем шрифты
    print("шрифты:")
    fonts = scan_fonts()
    print()

    # загружаем словарь
    print("словарь:")
    words = load_words()
    if not words:
        words = ["привет", "мир", "слово", "текст", "буква"]  # минимальный fallback
    print()

    all_chars   = C.RUS_LOWER + C.RUS_UPPER  # все буквы

    # ------------------------------------------------------------------
    # буквы
    # ------------------------------------------------------------------
    print("+--- Letters " + "-" * 48 + "+")
    print(f"  символов: {len(all_chars)}  |  на каждый: {C.LETTERS_PER_CHAR}")

    letter_tasks = []
    for idx, char in enumerate(all_chars * C.LETTERS_PER_CHAR):
        # перемешиваем чтобы не было все а потом все б
        letter_tasks.append((idx, char, C.LETTERS_DIR, C.CHAR_H, C.CHAR_W, "letter"))
    random.shuffle(letter_tasks)
    # переиндексируем после перемешивания
    letter_tasks = [(i,) + t[1:] for i, t in enumerate(letter_tasks)]

    generate_group(letter_tasks, C.LETTERS_DIR, "letters", fonts, words)
    print()

    # ------------------------------------------------------------------
    # цифры
    # ------------------------------------------------------------------
    print("+--- Nums " + "-" * 51 + "+")
    print(f"  символов: {len(C.DIGITS)}  |  на каждый: {C.NUMS_PER_CHAR}")

    num_tasks = []
    for char in C.DIGITS:
        for i in range(C.NUMS_PER_CHAR):
            num_tasks.append((0, char, C.NUMS_DIR, C.CHAR_H, C.CHAR_W, f"num_{char}"))
    random.shuffle(num_tasks)
    num_tasks = [(i,) + t[1:] for i, t in enumerate(num_tasks)]

    generate_group(num_tasks, C.NUMS_DIR, "nums", fonts, words)
    print()

    # ------------------------------------------------------------------
    # пунктуация
    # ------------------------------------------------------------------
    print("+--- Punkt " + "-" * 50 + "+")
    print(f"  символов: {len(C.PUNKT)}  |  на каждый: {C.PUNKT_PER_CHAR}")

    punkt_tasks = []
    for char in C.PUNKT:
        sname = SAFE_NAMES.get(char, f"U{ord(char):04X}")
        for i in range(C.PUNKT_PER_CHAR):
            punkt_tasks.append((0, char, C.PUNKT_DIR, C.CHAR_H, C.CHAR_W, sname))
    random.shuffle(punkt_tasks)
    punkt_tasks = [(i,) + t[1:] for i, t in enumerate(punkt_tasks)]

    generate_group(punkt_tasks, C.PUNKT_DIR, "punkt", fonts, words)
    print()

    # ------------------------------------------------------------------
    # слова
    # ------------------------------------------------------------------
    print("+--- Words " + "-" * 50 + "+")
    print(f"  слов: {C.WORDS_COUNT}")

    word_pool  = [w for w in words if 2 <= len(w) <= 20]
    word_tasks = []
    for i in range(C.WORDS_COUNT):
        word = random.choice(word_pool)
        word_tasks.append((i, word, C.WORDS_DIR, C.WORD_H, None, "word"))

    generate_group(word_tasks, C.WORDS_DIR, "words", fonts, words)
    print()

    # ------------------------------------------------------------------
    # предложения и абзацы
    # ------------------------------------------------------------------
    print("+--- Sentences " + "-" * 46 + "+")
    print(f"  предложений/абзацев: {C.SENTENCES_COUNT}")
    print(f"  размер изображения: {C.SENTENCE_W}x{C.SENTENCE_H}px")

    sentence_tasks = []
    for i in range(C.SENTENCES_COUNT):
        # половина - предложения половина - абзацы
        if i % 2 == 0:
            text = build_sentence(word_pool)
        else:
            text = build_paragraph(word_pool)
        sentence_tasks.append((i, text, C.SENTENCES_DIR, C.SENTENCE_H, C.SENTENCE_W, "sent"))

    generate_group(sentence_tasks, C.SENTENCES_DIR, "sentences", fonts, words)
    print()

    # итог
    print("+============================================================+")
    print("|  готово!                                                   |")
    print("+------------------------------------------------------------+")
    for folder, name in [
        (C.LETTERS_DIR,   "letters  "),
        (C.NUMS_DIR,      "nums     "),
        (C.PUNKT_DIR,     "punkt    "),
        (C.WORDS_DIR,     "words    "),
        (C.SENTENCES_DIR, "sentences"),
    ]:
        n = sum(1 for _ in Path(folder).glob("*.png"))
        print(f"|  {name}: {n:>8} изображений  ({folder})")
    print("+============================================================+")


if __name__ == "__main__":
    print("=" * 60)
    print("ОПТИМИЗАЦИИ ДЛЯ MACBOOK AIR M1:")
    print("• 16 параллельных процессов (8 ядер × 2)")
    print("• Spawn context для стабильности на macOS")
    print("• Быстрое сохранение PNG без сжатия")
    print("• Оптимизированный рендер текста")
    print("• 5000 задач на процесс для эффективности")
    print("=" * 60)
    for pkg in ("cv2", "PIL", "numpy", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"  установи: pip install opencv-python pillow numpy tqdm")
            sys.exit(1)
    main()