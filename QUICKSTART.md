# 🚀 БЫСТРЫЙ СТАРТ - G-VISION CNN Classifier

## ✅ Что изменилось

Переписан `train.py` с новой архитектурой:
- **Было:** YOLOv8l для детекции И классификации одновременно
- **Теперь:** YOLOv8 детектор + EfficientNet-B0 классификатор

## 📋 1. Подготовка

```bash
# Перейти в директорию проекта
cd /Users/garrisen/PycharmProjects/G-VISION

# Активировать venv
source .venv/bin/activate

# Обновить зависимости
pip install -r requirements.txt
```

## 🎯 2. Запуск обучения

```bash
python train.py
```

**Процесс:**
1. ✅ Проверка окружения + загрузка словаря
2. ✅ Сканирование датасета (буквы/цифры/знаки)
3. ✅ Подготовка CNN датасета (нормализация 64×64)
4. ✅ Обучение EfficientNet-B0 (~1-3 часа на маке с MPS)
5. ✅ Экспорт конфига
6. ✅ Тестирование на реальных фото

## 📊 3. Ожидаемые результаты

После обучения создадутся файлы:

```
g-vision-classifier.pt      ← 4-5 MB (обученная CNN)
g-vision-config.json         ← маппинг класов
validate_results/
  - results.txt              ← логи распознавания
  - *_annotated.jpg          ← визуализация
```

## ⚙️ 4. Настройка (если нужно)

Редактировать `Config` класс в `train.py` для:

```python
# Для быстрого теста (10 эпох вместо 100)
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 64  # если есть память

# Для точности (медленнее)
CNN_BATCH_SIZE = 16
LEARN_RATE = 0.0005
CNN_EPOCHS = 200
```

## 🔬 5. Использование после обучения

```python
from train import GVisionOCR

ocr = GVisionOCR(
    "best.pt",                        # детектор
    "g-vision-classifier.pt",         # классификатор
    "g-vision-config.json"            # конфиг
)

# Распознавание
result = ocr.recognize("test.jpg")
print(result["text"])
```

## 📚 Детали

- **Архитектура:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Основной код:** [train.py](train.py)
- **Зависимости:** [requirements.txt](requirements.txt)

## 🆘 Проблемы?

### GPU не распознается
```bash
python -c "import torch; print(torch.backends.mps.is_available())"  # M1/M2 Mac
```

### Недостаточно памяти
→ Уменьшить `CNN_BATCH_SIZE` в конфиге

### Очень медленное обучение
→ Увеличить `CNN_BATCH_SIZE` если есть память

---

✨ **Готово! Архитектура полностью обновлена и готова к обучению.**
