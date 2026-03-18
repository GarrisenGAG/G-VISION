#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
+==============================================================+
|        G-VISION OCR -- ФИНАЛЬНЫЙ ПАЙПЛАЙН ОБУЧЕНИЯ v4       |
|      YOLOv8l  |  Максимальная точность                       |
+==============================================================+

Структура датасета:
  DataSet/Train/
    letters/   <- папки по одной букве: а/ б/ А/ Б/ ...
    words/     <- labels.txt + изображения слов
    nums/      <- labels.txt + изображения цифр
    punkt/     <- labels.txt + изображения пунктуации

Запуск:
  python train_final.py

Результат:
  g-vision-final.pt   <- модель
  g-vision-config.json <- конфиг для инференса
"""

import os, sys, json, shutil, random, subprocess, urllib.request
from pathlib import Path
from collections import defaultdict


# ================================================================
#  ВСЕ НАСТРОЙКИ
# ================================================================
class Config:
    DATA_DIR = 'DataSet/Train' #путь к тренировочному дата-сету

    WORDS_DIR = 'DataSet/Train/Words' #путь к дате слов
    WORDS_LABELS = 'DataSet/Train/Words/labels.txt' #путь к лейблам даты слов

    NUM_DIR = 'DataSet/Train/Nums' #путь к дате цифр
    NUM_LABELS = 'DataSet/Train/Nums/labels.txt' #путь к лейблам даты цифр

    PUNCTUATION_DIR = 'DataSet/Train/punkt' #путь к дате пунктуации
    PUNCTUATION_LABELS = 'DataSet/Train/punkt/labels.txt' #путь к лейблам пунктуации

    VALIDATION_DIR = 'DataSet/Validate' #путь к валидационной дате

    #настройки модели
    BASE_MODEL = "yolov8l.pt"  # 43M параметров
    EPOCHS = 150 #- количество раз сколько нейросеть просмотрит весь дата сет
    PATIENCE = 25 #- если нейросеть больше 25 эпох не улучшает результаты, обучение заканчивается
    IMG_SIZE = 64  # Размер изображений
    BATCH = 16  # Показатель сколько изображений за 1 итерацию будут показаны нейросети
    WORKERS = 0  # Количество потоков загрузки данных с диска
    DEVICE = "0" # выбор видеокарты

    #оптимизация
    OPTIMIZER = "AdamW"
    LR0 = 0.0005
    LRF = 0.05
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 5
    CLOSE_MOSAIC = 15

    # -- Датасет -------------------------------------------------
    TRAIN_SPLIT   = 0.85
    VAL_SPLIT     = 0.10           # test = 5%

    # -- Словарь для коррекции -----------------------------------
    DICT_FILE     = "russian_words.txt"
    FUZZY_THR     = 0.78

    # -- TTA на инференсе ----------------------------------------
    USE_TTA       = True

    # -- Выходные файлы ------------------------------------------
    YOLO_DS       = "DataSet/YOLO"
    RUNS_DIR      = "runs/ocr"
    RUN_NAME      = "g-vision-l"
    FINAL_MODEL   = "g-vision-final.pt"
    FINAL_CONFIG  = "g-vision-config.json"


C = Config()

IMG_EXT     = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
PUNCT_CHARS = set('.,!?:;-—()[]«»"\'')


# ================================================================
#  ШАГ 0: Окружение
# ================================================================
def step0_check_env():
    _header("0", "Проверка окружения")

    pkgs = [
        ("ultralytics", "ultralytics", "ultralytics"),
        ("PyYAML",      "yaml",        "pyyaml"),
        ("opencv",      "cv2",         "opencv-python"),
        ("tqdm",        "tqdm",        "tqdm"),
    ]
    for name, imp, install in pkgs:
        try:
            mod = __import__(imp)
            _ok(f"{name} {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            _info(f"Устанавливаю {name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install, "-q"])
            _ok(f"{name} установлен")

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"CUDA  {name}  {mem:.1f} GB")
        _ok(f"BATCH = {C.BATCH}  (IMG={C.IMG_SIZE}x{C.IMG_SIZE})")
    else:
        _warn("GPU не найден — обучение на CPU будет очень медленным!")

    _ok(f"PyTorch {torch.__version__}")

    # Словарь
    if not Path(C.DICT_FILE).exists():
        _info("Скачиваю словарь...")
        url = "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt"
        try:
            urllib.request.urlretrieve(url, C.DICT_FILE)
            _ok(f"Словарь: {Path(C.DICT_FILE).stat().st_size // 1024} KB")
        except Exception as e:
            _warn(f"Словарь не скачался: {e}")
    else:
        _ok(f"Словарь: {C.DICT_FILE}")


# ================================================================
#  ШАГ 1: Сканирование всех четырёх источников
# ================================================================
def _scan_folders(base: Path, kind: str) -> list:
    """Структура: base / <символ> / image.png"""
    recs = []
    if not base.exists():
        _warn(f"Не найдено: {base}")
        return recs
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        char = folder.name
        for f in folder.iterdir():
            if f.suffix.lower() in IMG_EXT:
                recs.append({"path": f, "label": char, "kind": kind})
    return recs


def _scan_labels(base: Path, labels_file: str, kind: str) -> list:
    """Структура: base / labels.txt  формат: fname|текст"""
    recs = []
    lp   = base / labels_file
    if not base.exists() or not lp.exists():
        _warn(f"Не найдено: {lp}")
        return recs
    with open(lp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            fname, text = line.split("|", 1)
            p = base / fname
            if p.exists() and p.suffix.lower() in IMG_EXT:
                recs.append({"path": p, "label": text, "kind": kind})
    return recs


def step1_scan_datasets():
    _header("1", "Сканирование датасетов")
    records = []

    # 1. Буквы — labels.txt + изображения рядом
    r = _scan_labels(Path(C.LETTERS_DIR), "labels.txt", "letter")
    if not r:
        # Запасной вариант — старая структура с подпапками по букве
        r = _scan_folders(Path(C.LETTERS_DIR), "letter")
    _ok(f"Буквы:       {len(r):>9}")
    records += r

    # 2. Слова (labels.txt)
    r = _scan_labels(Path(C.WORDS_DIR), C.WORDS_LABELS, "word")
    _ok(f"Слова:       {len(r):>9}")
    records += r

    # 3. Цифры (labels.txt)
    r = _scan_labels(Path(C.NUMS_DIR), C.NUMS_LABELS, "num")
    _ok(f"Цифры:       {len(r):>9}")
    records += r

    # 4. Пунктуация (labels.txt)
    r = _scan_labels(Path(C.PUNKT_DIR), C.PUNKT_LABELS, "punct")
    _ok(f"Пунктуация:  {len(r):>9}")
    records += r

    if not records:
        _die("Датасет пуст! Проверь пути в Cfg.")

    # Классы = уникальные СИМВОЛЫ
    all_chars = set()
    for rec in records:
        for ch in rec["label"]:
            all_chars.add(ch)
    classes = sorted(all_chars)

    _ok(f"Итого:       {len(records):>9}  образцов  |  {len(classes)} классов")
    _info("Символы: " + "".join(classes[:80]) + ("..." if len(classes) > 80 else ""))
    return records, classes


# ================================================================
#  ШАГ 2: Конвертация через жёсткие ссылки (hardlinks)
#  Hardlink = файл занимает 0 дополнительных байт на диске,
#  создаётся мгновенно, YOLO видит его как обычный файл.
#  Fallback: если диски разные — обычное копирование.
# ================================================================
def _write_data_yaml(yaml_path: Path, root: Path, classes: list):
    import yaml
    root_str = str(root.resolve()).replace("\\", "/")
    data = {
        "path":  root_str,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),
        "names": classes,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    _ok(f"data.yaml: {len(classes)} классов")


def _link_or_copy(src: Path, dst: Path):
    """Жёсткая ссылка если на одном диске, иначе копия."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def step2_make_yolo_dataset(records: list, classes: list):
    _header("2", "Конвертация в YOLO-формат (hardlinks)")
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for i, c in enumerate(classes)}

    root     = Path(C.YOLO_DS)
    yaml_path = root / "data.yaml"

    # --- Пропускаем если датасет уже готов ---
    cache_train = root / "labels" / "train.cache"
    cache_val   = root / "labels" / "val.cache"
    if yaml_path.exists() and cache_train.exists() and cache_val.exists():
        _ok("Датасет уже существует + кэш найден — пропускаем шаг 2")
        _info("Чтобы пересобрать: удали папку DataSet\\YOLO и запусти снова")
        # Читаем маппинг из существующего файла
        mapping_path = root / "class_mapping.json"
        if mapping_path.exists():
            import json as _json
            m   = _json.loads(mapping_path.read_text(encoding="utf-8"))
            c2i = m["c2i"]
            i2c = {int(k): v for k, v in m["i2c"].items()}
        return str(yaml_path), c2i, i2c

    # Строим датасет с нуля
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    random.seed(42)
    random.shuffle(records)
    n       = len(records)
    n_train = int(n * C.TRAIN_SPLIT)
    n_val   = int(n * C.VAL_SPLIT)
    split_tags = (["train"] * n_train +
                  ["val"]   * n_val   +
                  ["test"]  * (n - n_train - n_val))

    name_count = defaultdict(int)
    tasks   = []
    skipped = 0

    _info("Подготовка списка файлов...")
    for rec, split in zip(records, split_tags):
        label      = rec["label"]
        first_char = label[0] if label else None
        if not first_char or first_char not in c2i:
            skipped += 1
            continue

        cls = c2i[first_char]
        src = rec["path"]

        name_count[src.stem] += 1
        cnt = name_count[src.stem]
        uid = src.stem if cnt == 1 else f"{src.stem}_{cnt}"

        dst_img  = root / "images" / split / (uid + src.suffix)
        lbl_file = root / "labels" / split / (uid + ".txt")

        tasks.append((src, dst_img, lbl_file, cls))

    if skipped:
        _warn(f"Пропущено (пустые метки): {skipped}")
    _ok(f"Задач: {len(tasks)}")

    # Проверяем поддержку hardlinks (одинаковый диск?)
    test_src = tasks[0][0] if tasks else None
    use_hardlinks = False
    if test_src:
        try:
            test_dst = root / "images" / "train" / "_hlink_test"
            os.link(test_src, test_dst)
            test_dst.unlink()
            use_hardlinks = True
        except OSError:
            use_hardlinks = False

    method = "жёсткие ссылки (0 байт доп. места)" if use_hardlinks else "копирование"
    _info(f"Метод: {method}")

    n_workers = min(32, (os.cpu_count() or 4) * 4)
    _info(f"Потоков: {n_workers}")

    def write_pair(t):
        src, dst_img, lbl_file, cls = t
        _link_or_copy(src, dst_img)
        lbl_file.write_text(f"{cls} 0.5 0.5 1.0 1.0\n")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(write_pair, t) for t in tasks]
        for _ in tqdm(as_completed(futs), total=len(futs),
                      desc="  Запись меток", ncols=75):
            pass

    _write_data_yaml(root / "data.yaml", root, classes)

    mapping = {
        "c2i":       c2i,
        "i2c":       {str(k): v for k, v in i2c.items()},
        "word_labels": list({r["label"] for r in records if r["kind"] == "word"})[:50000],
    }
    (root / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_tr = split_tags.count("train")
    n_va = split_tags.count("val")
    n_te = split_tags.count("test")
    _ok(f"Train: {n_tr}  |  Val: {n_va}  |  Test: {n_te}")
    return str(root / "data.yaml"), c2i, i2c


# ================================================================
#  ШАГ 3: Обучение YOLOv8l
# ================================================================
def step3_train(data_yaml: str):
    _header("3", f"Дообучение {C.BASE_MODEL}")
    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"VRAM: {mem_gb:.1f} GB  |  BATCH: {C.BATCH}  |  IMG: {C.IMG_SIZE}")

    model  = YOLO(C.BASE_MODEL)
    n_par  = sum(p.numel() for p in model.model.parameters())
    _ok(f"Параметров: {n_par:,}")

    results = model.train(
        data             = data_yaml,
        epochs           = C.EPOCHS,
        patience         = C.PATIENCE,
        batch            = C.BATCH,
        imgsz            = C.IMG_SIZE,
        device           = C.DEVICE,
        workers          = C.WORKERS,
        project          = C.RUNS_DIR,
        name             = C.RUN_NAME,
        exist_ok         = True,
        verbose          = True,
        save             = True,
        save_period      = 10,

        # Оптимизатор
        optimizer        = C.OPTIMIZER,
        lr0              = C.LR0,
        lrf              = C.LRF,
        momentum         = C.MOMENTUM,
        weight_decay     = C.WEIGHT_DECAY,
        warmup_epochs    = C.WARMUP_EPOCHS,
        warmup_momentum  = 0.5,
        warmup_bias_lr   = 0.05,
        close_mosaic     = C.CLOSE_MOSAIC,

        # Аугментации под рукописные символы
        hsv_h            = 0.0,   # ч/б — оттенок не трогаем
        hsv_s            = 0.3,
        hsv_v            = 0.5,
        degrees          = 12.0,  # наклон
        translate        = 0.1,
        scale            = 0.35,
        shear            = 8.0,
        perspective      = 0.0005,
        flipud           = 0.0,
        fliplr           = 0.0,   # НЕЛЬЗЯ зеркалить буквы!
        mosaic           = 0.5,
        mixup            = 0.15,
        copy_paste       = 0.0,
        erasing          = 0.35,

        # NMS
        conf             = 0.001,
        iou              = 0.6,
        max_det          = 300,
    )

    best = Path(C.RUNS_DIR) / C.RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        _die(f"best.pt не найден: {best}")

    # Метрики
    try:
        m = results.results_dict
        _ok(f"mAP50:     {m.get('metrics/mAP50(B)',    0):.4f}")
        _ok(f"mAP50-95:  {m.get('metrics/mAP50-95(B)',0):.4f}")
        _ok(f"Precision: {m.get('metrics/precision(B)',0):.4f}")
        _ok(f"Recall:    {m.get('metrics/recall(B)',   0):.4f}")
    except Exception:
        pass

    _ok(f"Лучшая модель: {best}")
    return str(best)


# ================================================================
#  ШАГ 4: Экспорт
# ================================================================
def step4_export(best_pt: str, c2i: dict, i2c: dict):
    _header("4", "Экспорт финальной модели")
    from ultralytics import YOLO

    shutil.copy2(best_pt, C.FINAL_MODEL)
    _ok(f"Модель: {C.FINAL_MODEL}")

    if C.USE_TTA:
        _info("TTA-валидация на test-сплите...")
        try:
            metrics = YOLO(C.FINAL_MODEL).val(
                data    = str(Path(C.YOLO_DS) / "data.yaml"),
                split   = "test",
                augment = True,
                verbose = False,
            )
            _ok(f"Test mAP50:    {metrics.box.map50:.4f}")
            _ok(f"Test mAP50-95: {metrics.box.map:.4f}")
        except Exception as e:
            _warn(f"TTA не удалась: {e}")

    config = {
        "model":     C.FINAL_MODEL,
        "base":      C.BASE_MODEL,
        "img_size":  C.IMG_SIZE,
        "use_tta":   C.USE_TTA,
        "c2i":       c2i,
        "i2c":       {str(k): v for k, v in i2c.items()},
        "dict_file": C.DICT_FILE,
        "fuzzy_thr": C.FUZZY_THR,
        "classes":   len(c2i),
    }
    Path(C.FINAL_CONFIG).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _ok(f"Конфиг: {C.FINAL_CONFIG}")
    return C.FINAL_MODEL, C.FINAL_CONFIG


# ================================================================
#  ЯЗЫКОВОЙ КОРРЕКТОР
# ================================================================
def _dlev(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            cost  = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                dp[j] = min(dp[j], prev[j-1])
    return dp[n]


class WordCorrector:
    def __init__(self, dict_file: str, thr: float):
        self.words = set()
        self.thr   = thr
        if os.path.exists(dict_file):
            with open(dict_file, encoding="utf-8") as f:
                self.words = {l.strip().lower() for l in f if l.strip()}

    def correct(self, word: str) -> str:
        if not word or not self.words:
            return word
        lw = word.lower()
        if lw in self.words:
            return word
        best, best_sim = word, 0.0
        for cand in self.words:
            if abs(len(cand) - len(lw)) > 3:
                continue
            sim = 1.0 - _dlev(lw, cand) / max(len(lw), len(cand), 1)
            if sim > best_sim:
                best_sim, best = sim, cand
        return best if best_sim >= self.thr else word


# ================================================================
#  КЛАСС ИНФЕРЕНСА
# ================================================================
class GVisionOCR:
    """
    Распознавание рукописного текста.

    Использование:
        ocr    = GVisionOCR("g-vision-final.pt", "g-vision-config.json")
        result = ocr.recognize("photo.jpg")
        print(result["text"])

    result содержит:
        text        — распознанный текст
        detections  — количество детекций
        letters     — количество букв/цифр
        punct       — количество знаков препинания
    """

    def __init__(self, model_path: str, config_path: str):
        import cv2
        from ultralytics import YOLO
        self.cv2   = cv2
        self.model = YOLO(model_path)

        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.i2c     = {int(k): v for k, v in cfg["i2c"].items()}
        self.corr    = WordCorrector(cfg.get("dict_file", ""),
                                     cfg.get("fuzzy_thr", 0.78))
        self.use_tta = cfg.get("use_tta", True)

    # 1. Предобработка
    def _preprocess(self, gray):
        cv2  = self.cv2
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, ot = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ad   = cv2.adaptiveThreshold(blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 8)
        comb = cv2.bitwise_or(ot, ad)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN,  k)
        return cv2.cvtColor(comb, cv2.COLOR_GRAY2BGR)

    # 2. Детекция
    def _detect(self, img_bgr):
        res  = self.model.predict(img_bgr, conf=0.25,
                                  augment=self.use_tta, verbose=False)[0]
        dets = []
        for box in res.boxes:
            cid   = int(box.cls[0])
            label = self.i2c.get(cid, "?")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append({
                "bbox":     (x1, y1, x2, y2),
                "label":    label,
                "conf":     float(box.conf[0]),
                "is_punct": all(c in PUNCT_CHARS for c in label),
            })
        return dets

    # 3. Коррекция
    def _fix(self, word: str) -> str:
        if not word or all(c in PUNCT_CHARS for c in word):
            return word
        up    = word[0].isupper()
        fixed = self.corr.correct(word)
        return fixed.capitalize() if up else fixed

    # 4. Сборка текста
    def _build_text(self, dets: list, img_h: int, img_w: int) -> str:
        if not dets:
            return ""
        dets = sorted(dets, key=lambda d: (d["bbox"][1], d["bbox"][0]))

        thr   = img_h * 0.05
        lines, cur = [], []
        for d in dets:
            yc = (d["bbox"][1] + d["bbox"][3]) / 2
            if not cur:
                cur.append(d)
            else:
                ly = (cur[-1]["bbox"][1] + cur[-1]["bbox"][3]) / 2
                if abs(yc - ly) <= thr:
                    cur.append(d)
                else:
                    lines.append(sorted(cur, key=lambda x: x["bbox"][0]))
                    cur = [d]
        if cur:
            lines.append(sorted(cur, key=lambda x: x["bbox"][0]))

        paragraphs, para = [], []
        for line in lines:
            tokens, buf, px2 = [], "", None
            for d in line:
                x1, _, x2, _ = d["bbox"]
                lbl = d["label"]
                gap = (x1 - px2) if px2 is not None else 0
                if px2 is not None and gap > 14:
                    if buf:
                        tokens.append(self._fix(buf))
                        buf = ""
                    if d["is_punct"]:
                        tokens.append(lbl)
                    else:
                        buf += lbl
                else:
                    if d["is_punct"]:
                        if buf:
                            tokens.append(self._fix(buf))
                            buf = ""
                        tokens.append(lbl)
                    else:
                        buf += lbl
                px2 = x2
            if buf:
                tokens.append(self._fix(buf))
            para.append(" ".join(tokens))
            line_w = sum(d["bbox"][2] - d["bbox"][0] for d in line)
            if line_w < img_w * 0.5:
                paragraphs.append(" ".join(para))
                para = []
        if para:
            paragraphs.append(" ".join(para))
        return "\n\n".join(paragraphs)

    def recognize(self, image_path: str, save_annotated: str = None) -> dict:
        img = self.cv2.imread(str(image_path))
        if img is None:
            return {"text": "", "error": f"Не читается: {image_path}"}

        gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        proc = self._preprocess(gray)
        dets = self._detect(proc)
        text = self._build_text(dets, h, w)

        if save_annotated:
            ann = img.copy()
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                c = (30, 210, 30) if not d["is_punct"] else (220, 100, 0)
                self.cv2.rectangle(ann, (x1, y1), (x2, y2), c, 1)
                self.cv2.putText(ann, d["label"], (x1, max(y1-4, 0)),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            self.cv2.imwrite(save_annotated, ann)

        return {
            "text":       text,
            "detections": len(dets),
            "letters":    sum(1 for d in dets if not d["is_punct"]),
            "punct":      sum(1 for d in dets if d["is_punct"]),
        }


# ================================================================
#  ШАГ 5: Тест на реальных изображениях из DataSet/Validate/
# ================================================================
def step5_test(final_model: str, final_config: str):
    _header("5", "Тест на реальных изображениях (Validate/)")

    validate_dir = Path(C.VALIDATE_DIR)
    if not validate_dir.exists():
        _warn(f"Папка не найдена: {validate_dir} — пропускаем")
        return

    # Собираем все изображения
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"):
        images.extend(validate_dir.glob(ext))
        images.extend(validate_dir.glob(ext.upper()))
    images = sorted(set(images))

    if not images:
        _warn(f"Нет изображений в {validate_dir} — пропускаем")
        return

    _ok(f"Найдено изображений: {len(images)}")

    # Папка для аннотированных результатов
    out_dir = Path("validate_results")
    out_dir.mkdir(exist_ok=True)
    _ok(f"Результаты → {out_dir.resolve()}")

    ocr = GVisionOCR(final_model, final_config)

    results_log = []
    total_dets  = 0
    failed      = 0

    print()
    sep = "-" * 62
    print(f"|  {sep}")
    print(f"|  {'Файл':<30} {'Символов':>8} {'Текст (превью)':<20}")
    print(f"|  {sep}")

    for img_path in images:
        ann_path = str(out_dir / (img_path.stem + "_annotated.jpg"))

        try:
            result = ocr.recognize(str(img_path), save_annotated=ann_path)
        except Exception as e:
            _warn(f"{img_path.name}: ошибка — {e}")
            failed += 1
            continue

        text    = result["text"]
        n_dets  = result["detections"]
        total_dets += n_dets

        # Превью текста для консоли (первые 25 символов)
        preview = text.replace("\n", " ")[:25]
        if len(text) > 25:
            preview += "..."

        print(f"|  {img_path.name:<30} {n_dets:>8}   {preview}")

        results_log.append({
            "file":       img_path.name,
            "text":       text,
            "detections": n_dets,
            "letters":    result["letters"],
            "punct":      result["punct"],
        })

    print(f"|  {sep}")
    print()

    # Сохраняем полный лог
    log_path = out_dir / "results.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results_log:
            f.write(f"=== {r['file']} ===\n")
            f.write(f"Детекций: {r['detections']}  "
                    f"(букв/цифр: {r['letters']}, знаков: {r['punct']})\n")
            f.write(f"Текст:\n{r['text']}\n\n")

    _ok(f"Обработано:   {len(results_log)}/{len(images)} изображений")
    _ok(f"Всего детекций: {total_dets}")
    if failed:
        _warn(f"Ошибок:       {failed}")
    _ok(f"Полный лог:   {log_path}")
    _ok(f"Аннотации:    {out_dir}/")


# ================================================================
#  ГЛАВНАЯ
# ================================================================
def main():
    print()
    print("+============================================================+")
    print("|     G-VISION OCR -- ФИНАЛЬНЫЙ ПАЙПЛАЙН ОБУЧЕНИЯ v4        |")
    print("|   YOLOv8l  |  letters + words + nums + punkt               |")
    print("+============================================================+")
    print()
    print(f"  Датасет:")
    print(f"    letters:  {C.LETTERS_DIR}")
    print(f"    words:    {C.WORDS_DIR}")
    print(f"    nums:     {C.NUMS_DIR}")
    print(f"    punkt:    {C.PUNKT_DIR}")
    print(f"    validate: {C.VALIDATE_DIR}")
    print()

    step0_check_env()
    records, classes          = step1_scan_datasets()
    data_yaml, c2i, i2c       = step2_make_yolo_dataset(records, classes)
    best_pt                   = step3_train(data_yaml)
    final_model, final_config = step4_export(best_pt, c2i, i2c)
    step5_test(final_model, final_config)

    print()
    print("+============================================================+")
    print("|  ГОТОВО!                                                   |")
    print(f"|  Модель:  {final_model:<51}|")
    print(f"|  Конфиг:  {final_config:<51}|")
    print("+------------------------------------------------------------+")
    print("|  Как использовать:                                         |")
    print("|                                                            |")
    print("|    from train_final import GVisionOCR                     |")
    print("|    ocr = GVisionOCR('g-vision-final.pt',                  |")
    print("|                     'g-vision-config.json')               |")
    print("|    result = ocr.recognize('photo.jpg')                    |")
    print("|    print(result['text'])                                   |")
    print("+============================================================+")


# ================================================================
#  Утилиты
# ================================================================
def _header(n, title):
    sep = "-" * max(0, 48 - len(title))
    print(); print(f"+--- ШАГ {n}: {title} {sep}+")

def _ok(msg):   print(f"|  OK    {msg}")
def _info(msg): print(f"|  ...   {msg}")
def _warn(msg): print(f"|  WARN  {msg}")
def _die(msg):
    print(f"\n  ОШИБКА: {msg}\n"); sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Прервано пользователем")
    except Exception as e:
        import traceback
        print(f"\n  Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)