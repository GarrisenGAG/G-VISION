import os,sys,json,shutil,random,subprocess,urllib.request
from pathlib import Path
from collections import defaultdict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os           # работа с файлами
import sys          # системные штуки - выход из программы и тд
import json         # чтение и запись json
import shutil       # копирование файлов
import random       # перемешивание датасета
import subprocess   # запуск команд прямо из кода - нужно для pip install
import urllib.request  # скачивание файлов из интернета
from pathlib import Path            # удобная работа с путями
from collections import defaultdict # словарь который сам создает дефолтное значение если ключа нет


# все настройки тут - менять только здесь
class Config:

    # пути к данным
    LETTERS_DIR    = "DataSet/Train/letters"  # папка с буквами
    LETTERS_LABELS = "labels.txt"             # файл разметки - формат: имя_файла|буква

    WORDS_DIR      = "DataSet/Train/words"    # папка со словами
    WORDS_LABELS   = "labels.txt"

    NUMS_DIR       = "DataSet/Train/nums"     # папка с цифрами
    NUMS_LABELS    = "labels.txt"

    PUNCT_DIR      = "DataSet/Train/punkt"    # пунктуация
    PUNCT_LABELS   = "labels.txt"

    VALIDATE_DIR   = "DataSet/Validate"       # реальные фотки для теста в конце

    # куда сохранять файлы
    YOLO_DATASET   = "DataSet/YOLO"           # датасет в формате который понимает yolo
    RUNS_DIR       = "runs/ocr"               # сюда yolo сохраняет логи и графики
    RUN_NAME       = "g-vision-l"             # название эксперимента
    FINAL_MODEL    = "g-vision-final.pt"      # итоговая модель
    FINAL_CONFIG   = "g-vision-config.json"   # конфиг - нужен чтобы знать что класс 42 = буква м

    # настройки модели
    BASE_MODEL     = "yolov8l.pt"  # базовая модель - скачается сама, 43 миллиона параметров
    EPOCHS         = 150           # сколько раз нейросеть пройдет по всему датасету
    PATIENCE       = 25            # если 25 эпох подряд нет улучшения - останавливаемся
    IMAGE_SIZE     = 64            # размер картинки на входе в пикселях
    BATCH_SIZE     = 16            # сколько картинок обрабатывается за одну итерацию
    WORKERS        = 0             # потоки загрузки данных - на windows обязательно 0
    GPU_INDEX      = "0"           # индекс видеокарты - 0 это первая карта

    # разбивка датасета
    TRAIN_PART     = 0.85   # 85% идет на обучение
    VAL_PART       = 0.10   # 10% на валидацию - проверка после каждой эпохи
                             # остаток 5% - финальный тест в самом конце

    # оптимизатор адамW - двигает веса модели в нужную сторону
    OPTIMIZER      = "AdamW"
    LEARN_RATE     = 0.0005   # начальная скорость обучения
    LEARN_RATE_END = 0.05     # коэффициент конечной скорости - lr умножается на него к концу
    MOMENTUM       = 0.937    # инерция - если градиент давит в одну сторону - ускоряемся
    WEIGHT_DECAY   = 0.0005   # штраф за большие веса - защита от переобучения
    WARMUP_EPOCHS  = 5        # первые 5 эпох скорость обучения плавно растет с нуля
    STOP_MOSAIC    = 15       # за 15 эпох до конца отключаем аугментацию mosaic

    # словарь для исправления ошибок распознавания
    DICT_FILE      = "russian_words.txt"  # скачается автоматически
    FUZZY_THRESH   = 0.78   # порог похожести - если слово совпадает на 78%+ то исправляем

    USE_TTA        = True   # при тесте делаем несколько предсказаний и усредняем - дает +1-3% точности


C = Config()  # создаем объект - теперь доступно как C.LETTERS_DIR и тд

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}  # допустимые форматы

PUNCT_SYMBOLS = set('.,!?:;-—()[]«»"\'')  # знаки пунктуации - нужны при сборке текста


# шаг 0 - проверяем что все установлено
def step0_check_environment():
    _header("0", "проверка окружения")

    # список нужных библиотек
    required_packages = [
        ("ultralytics", "ultralytics", "ultralytics"),  # сама yolo
        ("PyYAML",      "yaml",        "pyyaml"),        # запись yaml файлов
        ("opencv",      "cv2",         "opencv-python"), # обработка изображений
        ("tqdm",        "tqdm",        "tqdm"),           # прогресс бары
    ]

    for package_name, import_name, pip_name in required_packages:
        try:
            module = __import__(import_name)  # пробуем импортировать
            version = getattr(module, "__version__", "ok")
            _ok(f"{package_name} {version}")
        except ImportError:
            _info(f"устанавливаю {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
            _ok(f"{package_name} установлен")

    import torch

    if torch.cuda.is_available():
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # байты в гб
        _ok(f"gpu: {gpu_name}  {gpu_memory:.1f} gb vram")
        _ok(f"batch = {C.BATCH_SIZE}  |  image_size = {C.IMAGE_SIZE}x{C.IMAGE_SIZE}")
    else:
        _warn("gpu не найден - обучение на cpu будет в 50-100x медленнее")

    _ok(f"pytorch {torch.__version__}")

    # скачиваем словарь если его нет
    if not Path(C.DICT_FILE).exists():
        _info("скачиваю словарь русских слов...")
        dict_url = "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt"
        try:
            urllib.request.urlretrieve(dict_url, C.DICT_FILE)
            size_kb = Path(C.DICT_FILE).stat().st_size // 1024
            _ok(f"словарь скачан: {C.DICT_FILE} ({size_kb} kb)")
        except Exception as error:
            _warn(f"словарь не скачался: {error}")
    else:
        _ok(f"словарь уже есть: {C.DICT_FILE}")


# шаг 1 - читаем датасет из всех 4 папок

def _read_labels_file(folder: Path, labels_filename: str, data_type: str) -> list:
    # читает labels.txt и возвращает список записей
    # формат labels.txt - имя_файла.png|метка
    records     = []
    labels_path = folder / labels_filename

    if not folder.exists() or not labels_path.exists():
        _warn(f"не найдено: {labels_path}")
        return records

    with open(labels_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # убираем пробелы и переносы строк
            if "|" not in line:
                continue  # пустые строки пропускаем

            filename, label = line.split("|", 1)  # разбиваем на имя файла и метку
            image_path = folder / filename

            if image_path.exists() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "path":  image_path,  # полный путь к картинке
                    "label": label,       # что на картинке
                    "type":  data_type,   # откуда запись - нужно для статистики
                })

    return records


def _read_folders_structure(folder: Path, data_type: str) -> list:
    # читает старую структуру - folder/буква/картинка.png
    # используется если нет labels.txt
    records = []
    if not folder.exists():
        _warn(f"не найдено: {folder}")
        return records

    for subfolder in sorted(folder.iterdir()):
        if not subfolder.is_dir():
            continue
        char = subfolder.name  # имя папки = символ
        for image_file in subfolder.iterdir():
            if image_file.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"path": image_file, "label": char, "type": data_type})

    return records


def step1_scan_datasets():
    _header("1", "сканирование датасетов")
    all_records = []

    # буквы
    letter_records = _read_labels_file(Path(C.LETTERS_DIR), C.LETTERS_LABELS, "letter")
    if not letter_records:
        letter_records = _read_folders_structure(Path(C.LETTERS_DIR), "letter")  # запасной вариант
    _ok(f"буквы:       {len(letter_records):>9}")
    all_records += letter_records

    # слова
    word_records = _read_labels_file(Path(C.WORDS_DIR), C.WORDS_LABELS, "word")
    _ok(f"слова:       {len(word_records):>9}")
    all_records += word_records

    # цифры
    num_records = _read_labels_file(Path(C.NUMS_DIR), C.NUMS_LABELS, "num")
    _ok(f"цифры:       {len(num_records):>9}")
    all_records += num_records

    # пунктуация
    punct_records = _read_labels_file(Path(C.PUNCT_DIR), C.PUNCT_LABELS, "punct")
    _ok(f"пунктуация:  {len(punct_records):>9}")
    all_records += punct_records

    if not all_records:
        _die("датасет пуст - проверь пути в Config")

    # собираем уникальные символы - это и есть классы для yolo
    # слово привет добавляет символы п р и в е т - итого около 90 классов а не миллион слов
    unique_chars = set()
    for record in all_records:
        for char in record["label"]:
            unique_chars.add(char)

    classes = sorted(unique_chars)  # сортируем для одинакового порядка при каждом запуске

    _ok(f"итого:       {len(all_records):>9}  образцов  |  {len(classes)} классов")
    _info("символы: " + "".join(classes[:80]) + ("..." if len(classes) > 80 else ""))
    return all_records, classes


# шаг 2 - готовим датасет в формате yolo

def _save_data_yaml(yaml_path: Path, root_folder: Path, classes: list):
    # сохраняет главный конфиг датасета для yolo
    # через pyyaml - иначе кириллица и спецсимволы ломают синтаксис
    import yaml

    root_str = str(root_folder.resolve()).replace("\\", "/")  # yolo на windows хочет прямые слэши

    yaml_data = {
        "path":  root_str,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),   # nc = number of classes
        "names": classes,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    _ok(f"data.yaml сохранен: {len(classes)} классов")


def _create_link_or_copy(source: Path, destination: Path):
    # создает жесткую ссылку вместо копии - два имени на один файл - 0 доп места на диске
    # если диски разные - делаем обычную копию
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def step2_prepare_yolo_dataset(all_records: list, classes: list):
    _header("2", "подготовка датасета в формате yolo")
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    char_to_id = {char: idx for idx, char in enumerate(classes)}  # а -> 42
    id_to_char = {idx: char for idx, char in enumerate(classes)}  # 42 -> а

    dataset_root = Path(C.YOLO_DATASET)
    yaml_file    = dataset_root / "data.yaml"

    # пропускаем если датасет уже готов
    train_cache = dataset_root / "labels" / "train.cache"
    val_cache   = dataset_root / "labels" / "val.cache"

    if yaml_file.exists() and train_cache.exists() and val_cache.exists():
        _ok("датасет уже готов + кэш найден - пропускаем шаг 2")
        _info("чтобы пересобрать - удали папку DataSet\\YOLO и запусти снова")

        mapping_file = dataset_root / "class_mapping.json"
        if mapping_file.exists():
            import json as _json
            saved = _json.loads(mapping_file.read_text(encoding="utf-8"))
            char_to_id = saved["char_to_id"]
            id_to_char = {int(k): v for k, v in saved["id_to_char"].items()}

        return str(yaml_file), char_to_id, id_to_char

    # создаем структуру папок
    if dataset_root.exists():
        shutil.rmtree(dataset_root)  # удаляем старый если есть

    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True)
        (dataset_root / "labels" / split).mkdir(parents=True)

    # перемешиваем и делим на train val test
    random.seed(42)  # фиксируем случайность - одинаковое разбиение при каждом запуске
    random.shuffle(all_records)

    total       = len(all_records)
    train_count = int(total * C.TRAIN_PART)
    val_count   = int(total * C.VAL_PART)

    split_tags = (
        ["train"] * train_count +
        ["val"]   * val_count   +
        ["test"]  * (total - train_count - val_count)
    )

    filename_counter = defaultdict(int)  # счетчик дублирующихся имен файлов
    tasks   = []
    skipped = 0

    _info("подготовка списка файлов...")
    for record, split in zip(all_records, split_tags):
        label      = record["label"]
        first_char = label[0] if label else None  # берем первый символ как класс

        if not first_char or first_char not in char_to_id:
            skipped += 1
            continue

        class_id   = char_to_id[first_char]
        source_img = record["path"]

        # уникальное имя если одинаковые файлы из разных папок
        filename_counter[source_img.stem] += 1
        count       = filename_counter[source_img.stem]
        unique_name = source_img.stem if count == 1 else f"{source_img.stem}_{count}"

        dest_image = dataset_root / "images" / split / (unique_name + source_img.suffix)
        label_file = dataset_root / "labels" / split / (unique_name + ".txt")

        tasks.append((source_img, dest_image, label_file, class_id))

    if skipped:
        _warn(f"пропущено записей: {skipped}")
    _ok(f"файлов для обработки: {len(tasks)}")

    # проверяем поддержку hardlinks
    can_hardlink = False
    if tasks:
        try:
            test_file = dataset_root / "images" / "train" / "_test_link"
            os.link(tasks[0][0], test_file)
            test_file.unlink()
            can_hardlink = True
        except OSError:
            can_hardlink = False

    method = "жесткие ссылки - 0 доп места" if can_hardlink else "копирование файлов"
    _info(f"метод: {method}")

    num_threads = min(32, (os.cpu_count() or 4) * 4)
    _info(f"потоков: {num_threads}")

    def process_one_file(task):
        src_img, dst_img, lbl_file, cls_id = task
        _create_link_or_copy(src_img, dst_img)
        # формат yolo - class_id x_center y_center width height - всё в долях от 1.0
        # 0.5 0.5 1.0 1.0 = объект занимает весь кадр - у нас один символ = весь кадр
        lbl_file.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_one_file, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="  запись файлов", ncols=75):
            pass

    _save_data_yaml(dataset_root / "data.yaml", dataset_root, classes)

    # сохраняем маппинг символов - без него модель не знает что класс 42 = буква м
    mapping_data = {
        "char_to_id": char_to_id,
        "id_to_char": {str(k): v for k, v in id_to_char.items()},
        "vocabulary": list({r["label"] for r in all_records if r["type"] == "word"})[:50000],
    }
    (dataset_root / "class_mapping.json").write_text(
        json.dumps(mapping_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _ok(f"train: {split_tags.count('train')}  |  val: {split_tags.count('val')}  |  test: {split_tags.count('test')}")
    return str(dataset_root / "data.yaml"), char_to_id, id_to_char


# шаг 3 - запускаем обучение
def step3_train_model(yaml_path: str):
    _header("3", f"обучение модели {C.BASE_MODEL}")
    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"vram: {gpu_memory:.1f} gb  |  batch: {C.BATCH_SIZE}  |  img: {C.IMAGE_SIZE}px")

    # загружаем предобученную модель - она уже умеет видеть формы и границы
    # нам остается только дообучить ее под русские символы
    model       = YOLO(C.BASE_MODEL)
    param_count = sum(p.numel() for p in model.model.parameters())
    _ok(f"параметров в модели: {param_count:,}")

    training_results = model.train(
        data        = yaml_path,      # путь к data.yaml
        epochs      = C.EPOCHS,       # количество эпох
        patience    = C.PATIENCE,     # early stopping - стоп если нет улучшения
        batch       = C.BATCH_SIZE,   # размер батча
        imgsz       = C.IMAGE_SIZE,   # размер входных картинок
        device      = C.GPU_INDEX,    # индекс видеокарты
        workers     = C.WORKERS,      # потоки загрузки - 0 на windows
        project     = C.RUNS_DIR,     # папка для логов
        name        = C.RUN_NAME,     # название эксперимента
        exist_ok    = True,           # не ругаться если папка уже есть
        verbose     = True,
        save        = True,
        save_period = 10,             # сохранять чекпоинт каждые 10 эпох
        cache       = "disk",         # кэш на диск - ускоряет следующие эпохи
        half        = True,           # fp16 вместо fp32 - в 2x быстрее на rtx tensor cores

        # оптимизатор
        optimizer     = C.OPTIMIZER,
        lr0           = C.LEARN_RATE,      # начальная скорость обучения
        lrf           = C.LEARN_RATE_END,  # финальная lr = lr0 * lrf
        momentum      = C.MOMENTUM,        # инерция beta1 в adamw
        weight_decay  = C.WEIGHT_DECAY,    # штраф за большие веса
        warmup_epochs = C.WARMUP_EPOCHS,   # эпохи прогрева
        warmup_momentum = 0.5,             # инерция во время прогрева
        warmup_bias_lr  = 0.05,            # скорость bias весов при прогреве
        close_mosaic    = C.STOP_MOSAIC,   # за n эпох до конца выключаем mosaic

        # аугментации - искусственно разнообразим датасет
        hsv_h       = 0.0,    # оттенок не меняем - текст чб
        hsv_s       = 0.3,    # насыщенность +-30%
        hsv_v       = 0.5,    # яркость +-50% - разное освещение при сканировании
        degrees     = 12.0,   # поворот +-12 градусов - рукопись под разным углом
        translate   = 0.1,    # сдвиг +-10%
        scale       = 0.35,   # масштаб +-35% - разный размер букв у разных людей
        shear       = 8.0,    # наклон +-8 градусов - имитация курсива
        perspective = 0.0005, # перспективное искажение - съемка под углом
        flipud      = 0.0,    # вертикальное отражение нельзя - п и ш станут похожи
        fliplr      = 0.0,    # горизонтальное нельзя - б и д станут похожи
        mosaic      = 0.5,    # склеиваем 4 картинки в одну
        mixup       = 0.15,   # смешиваем 2 картинки полупрозрачно
        copy_paste  = 0.0,    # не нужно для символов
        erasing     = 0.35,   # стираем 35% пикселей - учим по неполным символам

        # nms - фильтрация дублирующихся рамок
        conf    = 0.001,  # очень низкий порог при обучении - не пропустить ничего
        iou     = 0.6,    # если два bbox перекрываются больше 60% - оставить один
        max_det = 300,    # максимум детекций на одно изображение
    )

    best_model_path = Path(C.RUNS_DIR) / C.RUN_NAME / "weights" / "best.pt"
    if not best_model_path.exists():
        _die(f"файл best.pt не найден: {best_model_path}")

    # выводим метрики
    try:
        metrics = training_results.results_dict
        _ok(f"mAP50:     {metrics.get('metrics/mAP50(B)',    0):.4f}  - основная метрика")
        _ok(f"mAP50-95:  {metrics.get('metrics/mAP50-95(B)',0):.4f}")
        _ok(f"precision: {metrics.get('metrics/precision(B)',0):.4f}  - точность")
        _ok(f"recall:    {metrics.get('metrics/recall(B)',   0):.4f}  - полнота")
    except Exception:
        pass

    _ok(f"лучшая модель: {best_model_path}")
    return str(best_model_path)


# шаг 4 - сохраняем финальную модель
def step4_export_model(best_model_path: str, char_to_id: dict, id_to_char: dict):
    _header("4", "экспорт финальной модели")
    from ultralytics import YOLO

    shutil.copy2(best_model_path, C.FINAL_MODEL)
    _ok(f"модель: {C.FINAL_MODEL}")

    if C.USE_TTA:
        _info("tta-валидация на test-сплите...")
        try:
            test_metrics = YOLO(C.FINAL_MODEL).val(
                data    = str(Path(C.YOLO_DATASET) / "data.yaml"),
                split   = "test",    # используем отложенные 5%
                augment = True,      # несколько предсказаний с аугментацией
                verbose = False,
            )
            _ok(f"test mAP50:    {test_metrics.box.map50:.4f}")
            _ok(f"test mAP50-95: {test_metrics.box.map:.4f}")
        except Exception as error:
            _warn(f"tta не удалась: {error}")

    # сохраняем конфиг - без него модель не знает что класс 42 это буква м
    config_data = {
        "model":        C.FINAL_MODEL,
        "base_model":   C.BASE_MODEL,
        "image_size":   C.IMAGE_SIZE,
        "use_tta":      C.USE_TTA,
        "char_to_id":   char_to_id,
        "id_to_char":   {str(k): v for k, v in id_to_char.items()},
        "dict_file":    C.DICT_FILE,
        "fuzzy_thresh": C.FUZZY_THRESH,
        "num_classes":  len(char_to_id),
    }

    Path(C.FINAL_CONFIG).write_text(
        json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _ok(f"конфиг: {C.FINAL_CONFIG}")
    return C.FINAL_MODEL, C.FINAL_CONFIG


# языковой корректор - исправляет ошибки распознавания через словарь

def _damerau_levenshtein(word_a: str, word_b: str) -> int:
    # считает минимальное количество операций чтобы превратить одно слово в другое
    # операции - вставка удаление замена перестановка двух соседних букв
    m, n = len(word_a), len(word_b)
    dp   = list(range(n + 1))

    for i in range(1, m + 1):
        prev_row, dp[0] = dp[:], i
        for j in range(1, n + 1):
            cost  = 0 if word_a[i-1] == word_b[j-1] else 1
            dp[j] = min(
                dp[j-1]       + 1,    # вставка
                prev_row[j]   + 1,    # удаление
                prev_row[j-1] + cost  # замена
            )
            if i > 1 and j > 1 and word_a[i-1] == word_b[j-2] and word_a[i-2] == word_b[j-1]:
                dp[j] = min(dp[j], prev_row[j-1])  # перестановка соседних букв

    return dp[n]


class WordCorrector:
    # исправляет неправильно распознанные слова используя словарь
    # если слово похоже на словарное на 78%+ - заменяем

    def __init__(self, dict_file: str, similarity_threshold: float):
        self.dictionary = set()
        self.threshold  = similarity_threshold

        if os.path.exists(dict_file):
            with open(dict_file, encoding="utf-8") as f:
                self.dictionary = {line.strip().lower() for line in f if line.strip()}

    def fix_word(self, word: str) -> str:
        if not word or not self.dictionary:
            return word

        word_lower = word.lower()

        if word_lower in self.dictionary:
            return word  # уже правильное - не трогаем

        best_match      = word
        best_similarity = 0.0

        for candidate in self.dictionary:
            if abs(len(candidate) - len(word_lower)) > 3:
                continue  # слишком разная длина - пропускаем

            distance   = _damerau_levenshtein(word_lower, candidate)
            similarity = 1.0 - distance / max(len(word_lower), len(candidate), 1)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match      = candidate

        if best_similarity >= self.threshold:
            return best_match.capitalize() if word[0].isupper() else best_match
        else:
            return word  # ничего похожего не нашли - оставляем как есть


# класс для распознавания текста - используется после обучения
class GVisionOCR:
    # распознает рукописный русский текст на фотографии
    #
    # использование:
    #   ocr    = GVisionOCR("g-vision-final.pt", "g-vision-config.json")
    #   result = ocr.recognize("фото.jpg")
    #   print(result["text"])

    def __init__(self, model_path: str, config_path: str):
        import cv2
        from ultralytics import YOLO

        self.cv2   = cv2
        self.model = YOLO(model_path)

        config        = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.id2char  = {int(k): v for k, v in config["id_to_char"].items()}
        self.corrector = WordCorrector(
            config.get("dict_file", ""),
            config.get("fuzzy_thresh", 0.78)
        )
        self.use_tta = config.get("use_tta", True)

    def _preprocess_image(self, gray_image):
        # подготавливает изображение - отделяет текст от фона
        cv2 = self.cv2

        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)  # убираем мелкий шум

        # otsu - автоматически находит глобальный порог бинаризации
        _, otsu_result = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # адаптивный threshold - работает с неравномерным освещением
        adaptive_result = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            15, 8
        )

        combined = cv2.bitwise_or(otsu_result, adaptive_result)  # берем лучшее от обоих

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)  # закрываем дырки в буквах
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)  # убираем пиксели-шум

        return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)  # yolo ожидает цветное изображение

    def _detect_symbols(self, bgr_image):
        # запускает yolo и возвращает список найденных символов с координатами
        yolo_result = self.model.predict(
            bgr_image,
            conf    = 0.25,
            augment = self.use_tta,
            verbose = False
        )[0]

        detections = []
        for box in yolo_result.boxes:
            class_id = int(box.cls[0])
            symbol   = self.id2char.get(class_id, "?")  # id -> символ
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "symbol":     symbol,
                "confidence": float(box.conf[0]),
                "is_punct":   all(c in PUNCT_SYMBOLS for c in symbol),
            })

        return detections

    def _fix_word(self, word: str) -> str:
        if not word or all(c in PUNCT_SYMBOLS for c in word):
            return word  # пунктуацию не трогаем
        is_capitalized = word[0].isupper()
        fixed = self.corrector.fix_word(word)
        return fixed.capitalize() if is_capitalized else fixed

    def _build_text(self, detections: list, img_h: int, img_w: int) -> str:
        # собирает текст из детекций
        # группирует символы в строки по y-координате потом строки в абзацы
        if not detections:
            return ""

        detections = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))

        line_threshold = img_h * 0.05  # 5% высоты - допустимый разброс y для одной строки
        lines          = []
        current_line   = []

        for det in detections:
            y_center = (det["bbox"][1] + det["bbox"][3]) / 2

            if not current_line:
                current_line.append(det)
            else:
                prev_y = (current_line[-1]["bbox"][1] + current_line[-1]["bbox"][3]) / 2
                if abs(y_center - prev_y) <= line_threshold:
                    current_line.append(det)
                else:
                    lines.append(sorted(current_line, key=lambda x: x["bbox"][0]))
                    current_line = [det]

        if current_line:
            lines.append(sorted(current_line, key=lambda x: x["bbox"][0]))

        paragraphs   = []
        current_para = []

        for line in lines:
            tokens       = []
            word_buffer  = ""
            prev_x_right = None

            for det in line:
                x_left, _, x_right, _ = det["bbox"]
                symbol = det["symbol"]
                gap = (x_left - prev_x_right) if prev_x_right is not None else 0

                if prev_x_right is not None and gap > 14:  # 14px - граница слова
                    if word_buffer:
                        tokens.append(self._fix_word(word_buffer))
                        word_buffer = ""
                    if det["is_punct"]:
                        tokens.append(symbol)
                    else:
                        word_buffer += symbol
                else:
                    if det["is_punct"]:
                        if word_buffer:
                            tokens.append(self._fix_word(word_buffer))
                            word_buffer = ""
                        tokens.append(symbol)
                    else:
                        word_buffer += symbol

                prev_x_right = x_right

            if word_buffer:
                tokens.append(self._fix_word(word_buffer))

            current_para.append(" ".join(tokens))

            line_width = sum(d["bbox"][2] - d["bbox"][0] for d in line)
            if line_width < img_w * 0.5:  # короткая строка - конец абзаца
                paragraphs.append(" ".join(current_para))
                current_para = []

        if current_para:
            paragraphs.append(" ".join(current_para))

        return "\n\n".join(paragraphs)

    def recognize(self, image_path: str, save_annotated: str = None) -> dict:
        image = self.cv2.imread(str(image_path))
        if image is None:
            return {"text": "", "error": f"не могу прочитать файл: {image_path}"}

        gray         = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape

        processed  = self._preprocess_image(gray)            # 1 - подготовка
        detections = self._detect_symbols(processed)          # 2 - детекция
        text       = self._build_text(detections, img_h, img_w)  # 3 - сборка текста

        if save_annotated:
            annotated = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                color = (30, 210, 30) if not det["is_punct"] else (220, 100, 0)  # зеленый или оранжевый
                self.cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                self.cv2.putText(annotated, det["symbol"], (x1, max(y1-4, 0)),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            self.cv2.imwrite(save_annotated, annotated)

        return {
            "text":       text,
            "detections": len(detections),
            "letters":    sum(1 for d in detections if not d["is_punct"]),
            "punct":      sum(1 for d in detections if d["is_punct"]),
        }


# шаг 5 - тест на реальных фотках из DataSet/Validate/
def step5_test_on_real_photos(final_model: str, final_config: str):
    _header("5", "тест на реальных фотографиях (Validate/)")

    validate_folder = Path(C.VALIDATE_DIR)
    if not validate_folder.exists():
        _warn(f"папка не найдена: {validate_folder} - пропускаем тест")
        return

    photos = []
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"):
        photos.extend(validate_folder.glob(extension))
        photos.extend(validate_folder.glob(extension.upper()))
    photos = sorted(set(photos))

    if not photos:
        _warn(f"нет изображений в {validate_folder}")
        return

    _ok(f"найдено фотографий: {len(photos)}")

    results_folder = Path("validate_results")
    results_folder.mkdir(exist_ok=True)
    _ok(f"результаты в: {results_folder.resolve()}")

    ocr = GVisionOCR(final_model, final_config)

    results_log = []
    total_found = 0
    errors      = 0

    separator = "-" * 62
    print(f"\n|  {separator}")
    print(f"|  {'файл':<30} {'символов':>8}   {'текст (превью)'}")
    print(f"|  {separator}")

    for photo_path in photos:
        annotated_path = str(results_folder / (photo_path.stem + "_annotated.jpg"))

        try:
            result = ocr.recognize(str(photo_path), save_annotated=annotated_path)
        except Exception as error:
            _warn(f"{photo_path.name}: ошибка - {error}")
            errors += 1
            continue

        recognized_text = result["text"]
        symbol_count    = result["detections"]
        total_found    += symbol_count

        text_preview = recognized_text.replace("\n", " ")[:25]
        if len(recognized_text) > 25:
            text_preview += "..."

        print(f"|  {photo_path.name:<30} {symbol_count:>8}   {text_preview}")

        results_log.append({
            "file":    photo_path.name,
            "text":    recognized_text,
            "symbols": symbol_count,
            "letters": result["letters"],
            "punct":   result["punct"],
        })

    print(f"|  {separator}\n")

    log_file = results_folder / "results.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        for r in results_log:
            f.write(f"=== {r['file']} ===\n")
            f.write(f"символов: {r['symbols']} (букв: {r['letters']}, знаков: {r['punct']})\n")
            f.write(f"текст:\n{r['text']}\n\n")

    _ok(f"обработано: {len(results_log)}/{len(photos)} фотографий")
    _ok(f"символов найдено всего: {total_found}")
    if errors:
        _warn(f"ошибок: {errors}")
    _ok(f"лог: {log_file}")
    _ok(f"аннотированные фото: {results_folder}/")


# точка входа - запускает все шаги по порядку
def main():
    print("  источники данных:")
    print(f"буквы:       {C.LETTERS_DIR}")
    print(f"слова:       {C.WORDS_DIR}")
    print(f"цифры:       {C.NUMS_DIR}")
    print(f"пунктуация:  {C.PUNCT_DIR}")
    print(f"тест:        {C.VALIDATE_DIR}")

    step0_check_environment()

    all_records, classes = step1_scan_datasets()

    yaml_path, char_to_id, id_to_char = step2_prepare_yolo_dataset(all_records, classes)

    best_model = step3_train_model(yaml_path)

    final_model, final_config = step4_export_model(best_model, char_to_id, id_to_char)

    step5_test_on_real_photos(final_model, final_config)

    print(f"модель:  {final_model:<51}|")
    print(f"конфиг:  {final_config:<51}|")


# вспомогательные функции для вывода
def _header(step_number, title):
    separator = "-" * max(0, 48 - len(title))
    print()
    print(f"+--- шаг {step_number}: {title} {separator}+")

def _ok(message):    print(f"|  ok    {message}")
def _info(message):  print(f"|  ...   {message}")
def _warn(message):  print(f"|  warn  {message}")

def _die(message):
    print(f"\n  ошибка: {message}\n")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  прервано пользователем")
    except Exception as error:
        import traceback
        print(f"\n  ошибка: {error}")
        traceback.print_exc()
        sys.exit(1)


