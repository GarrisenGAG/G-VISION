"""G-Vision OCR - macOS Version"""
import sys
import json
import customtkinter as ctk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import threading
import logging
import os
import cv2
import numpy as np

# Константы
__version__ = "1.0.0"
APP_NAME = "G-Vision OCR"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
MIN_WIDTH = 700
MIN_HEIGHT = 500
IMAGE_PREVIEW_WIDTH = 600
IMAGE_PREVIEW_HEIGHT = 400

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnimationIndicator:
    """Маленький анимированный индикатор загрузки с прозрачностью"""
    def __init__(self, video_path, size=48):
        self.video_path = video_path
        self.size = size
        self.frames = []
        self.current_frame_index = 0
        self._load_frames()
    
    def _load_frames(self):
        """Загрузка и масштабирование кадров"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {self.video_path}")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Преобразование BGR в RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Масштабирование до нужного размера
                frame_resized = cv2.resize(frame, (self.size, self.size))
                
                # Преобразование в PIL Image
                pil_image = Image.fromarray(frame_resized).convert("RGBA")

                # Преобразуем черный фон в прозрачный и задаем общую непрозрачность
                arr = np.array(pil_image)
                # Черный цвет (либо очень темный) становится полностью прозрачным
                black_mask = (arr[:, :, 0] < 30) & (arr[:, :, 1] < 30) & (arr[:, :, 2] < 30)
                arr[black_mask, 3] = 0
                # Все остальное — на уровне 78% (200)
                arr[~black_mask, 3] = 200
                pil_image = Image.fromarray(arr, mode='RGBA')

                self.frames.append(pil_image)
            
            cap.release()
            logger.info(f"Loaded {len(self.frames)} indicator frames from {self.video_path}")
        except Exception as e:
            logger.error(f"Error loading frames: {e}")
    
    def get_frame(self, index):
        """Получить кадр по индексу"""
        if len(self.frames) == 0:
            return None
        
        # Циклическое проигрывание
        index = index % len(self.frames)
        return self.frames[index]
    
    def get_frames_count(self):
        """Получить количество кадров"""
        return len(self.frames)
    
    def reset_index(self):
        """Сбросить индекс"""
        self.current_frame_index = 0


class App(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            self.title(APP_NAME)
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
            self.minsize(MIN_WIDTH, MIN_HEIGHT)

            # Центрирование окна на экране
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - WINDOW_WIDTH) // 2
            y = (screen_h - WINDOW_HEIGHT) // 2
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

            self.configure(fg_color="#010101")

            # Инициализация анимации для индикатора
            self.animation_indicator = None
            self.animation_index = 0
            self.is_showing_animation = False
            self._init_animation_indicator()

            self.main_container = ctk.CTkFrame(self, fg_color="transparent")
            self.main_container.pack(fill="both", expand=True, padx=15, pady=(15, 15))

            self.sidebar = ctk.CTkFrame(
                self.main_container,
                fg_color="#0f1638",
                corner_radius=16,
                width=220
            )
            self.sidebar.pack(side="left", fill="y", padx=(0, 10), pady=0)
            self.sidebar.pack_propagate(False)

            self.content_area = ctk.CTkFrame(
                self.main_container,
                fg_color="#0a0f24",
                corner_radius=16
            )
            self.content_area.pack(side="right", fill="both", expand=True)

            # Инициализация состояния
            self.image_path = None
            self.is_processing = False
            self.current_image = None
            self.ocr = None

            self._build_sidebar()
            self._build_content()
            self._setup_shortcuts()

            # Попытка подгрузить OCR-модель
            self._init_ocr()

            # Начальный эффект плавного появления окна
            self.attributes("-alpha", 0.0)
            self._fade_window_to(target_alpha=1.0, duration=300, steps=20)

            logger.info("App initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing app: {e}", exc_info=True)
            raise

    def _fade_in_window(self, steps=20, interval=15):
        """Плавное появление окна при запуске"""
        def step(i):
            value = min(1.0, i / steps)
            self.attributes("-alpha", value)
            if i < steps:
                self.after(interval, lambda: step(i + 1))
        step(0)

    def _animate_frame_height(self, target_height, duration=220, steps=10):
        """Плавное изменение высоты image_frame"""
        start_height = max(1, self.image_frame.winfo_height() or 280)
        delta = target_height - start_height
        if delta == 0:
            return

        def step(i):
            new_h = int(start_height + delta * (i / steps))
            self.image_frame.configure(height=new_h)
            if i < steps:
                self.after(int(duration / steps), lambda: step(i + 1))

        step(1)

    def _pulse_image_frame(self, count=4, interval=80):
        """Пульсирующее подсвечивание рамки image_frame"""
        colors = ["#0a1020", "#1d426c", "#0a1020"]

        def step(i):
            self.image_frame.configure(fg_color=colors[i % len(colors)])
            if i < count:
                self.after(interval, lambda: step(i + 1))
            else:
                self.image_frame.configure(fg_color="#0a1020")

        step(0)

    def _fade_window_to(self, target_alpha=1.0, duration=240, steps=10):
        """Плавное изменение прозрачности окна"""
        try:
            start_alpha = float(self.attributes("-alpha") or 1.0)
        except Exception:
            start_alpha = 1.0
        delta = target_alpha - start_alpha

        def step(i):
            self.attributes("-alpha", start_alpha + delta * (i / steps))
            if i < steps:
                self.after(int(duration / steps), lambda: step(i + 1))

        step(1)

    def _init_animation_indicator(self):
        """Инициализация маленького анимированного индикатора"""
        try:
            video_path = os.path.join(os.path.dirname(__file__), "0001-0048.mkv")
            if os.path.exists(video_path):
                self.animation_indicator = AnimationIndicator(video_path, size=32)
                logger.info(f"Animation indicator initialized with {self.animation_indicator.get_frames_count()} frames")
            else:
                logger.warning(f"Animation file not found: {video_path}")
        except Exception as e:
            logger.error(f"Error initializing animation indicator: {e}")

    def _init_ocr(self):
        """Подключение предварительно обученной модели OCR best.pt"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            config_path = os.path.join(os.path.dirname(__file__), "g-vision-config.json")

            if not os.path.exists(model_path):
                self._set_status("Модель best.pt не найдена", "#e74c3c")
                logger.warning(f"OCR model not found: {model_path}")
                return

            # Если конфиг отсутствует, пробуем сгенерировать из class_mapping (train.py)
            if not os.path.exists(config_path):
                fallback_mapping = os.path.join(os.path.dirname(__file__), "Data-set/YOLO/class_mapping.json")
                if os.path.exists(fallback_mapping):
                    try:
                        with open(fallback_mapping, encoding="utf-8") as f:
                            mapping_json = json.load(f)
                        config_data = {
                            "model": model_path,
                            "base_model": "yolov8l.pt",
                            "image_size": 64,
                            "use_tta": True,
                            "char_to_id": mapping_json.get("char_to_id", {}),
                            "id_to_char": mapping_json.get("id_to_char", {}),
                            "dict_file": "russian_words.txt",
                            "fuzzy_thresh": 0.78,
                            "num_classes": len(mapping_json.get("char_to_id", {})),
                        }
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(config_data, f, ensure_ascii=False, indent=2)
                        logger.info(f"Created fallback config: {config_path}")
                        self._set_status("Создан fallback config для OCR", "#f1c40f")
                    except Exception as ex:
                        logger.warning(f"Не удалось создать fallback config: {ex}")

            if not os.path.exists(config_path):
                self._set_status("Конфиг g-vision-config.json не найден", "#e74c3c")
                logger.warning(f"OCR config not found: {config_path}")
                return

            from train import GVisionOCR
            self.ocr = GVisionOCR(model_path, config_path)
            self._set_status("Модель best.pt подключена", "#2ecc71")
            logger.info(f"OCR model initialized from {model_path}, config {config_path}")
        except Exception as e:
            logger.error(f"Error initializing OCR model: {e}", exc_info=True)
            self.ocr = None
            self._set_status("Ошибка при загрузке OCR", "#e74c3c")

    def _setup_shortcuts(self):
        """Настройка горячих клавиш"""
        shortcuts = [
            ('<Escape>', lambda e: self.destroy()),
            ('<Control-o>', lambda e: self.load_image()),
            ('<Command-o>', lambda e: self.load_image()),  # macOS
            ('<Control-r>', lambda e: self.recognize()),
            ('<Command-r>', lambda e: self.recognize()),  # macOS
            ('<Control-l>', lambda e: self.clear()),
            ('<Command-l>', lambda e: self.clear()),  # macOS
        ]
        for key, command in shortcuts:
            self.bind(key, command)

    def _build_sidebar(self):
        title = ctk.CTkLabel(
            self.sidebar, text="Управление",
            font=ctk.CTkFont(family="SF Pro Display", size=18, weight="bold"),
            text_color="#ffffff"
        )
        title.pack(pady=(20, 25), padx=15)

        btn_style = {
            "font": ctk.CTkFont(family="SF Pro Text", size=14, weight="bold"),
            "height": 44,
            "corner_radius": 10,
            "fg_color": "#1a2a5a",
            "hover_color": "#243a7a",
            "text_color": "#ffffff",
        }

        self.btn_load = ctk.CTkButton(self.sidebar, text="📁 Загрузить фото", command=self.load_image, **btn_style)
        self.btn_load.pack(pady=6, padx=15, fill="x")

        self.btn_run = ctk.CTkButton(self.sidebar, text="🔍 Распознать", command=self.recognize, **btn_style)
        self.btn_run.pack(pady=6, padx=15, fill="x")

        self.btn_clear = ctk.CTkButton(self.sidebar, text="🗑️ Очистить", command=self.clear, **btn_style)
        self.btn_clear.pack(pady=6, padx=15, fill="x")

        separator = ctk.CTkFrame(self.sidebar, height=1, fg_color="#2a3a6a")
        separator.pack(fill="x", padx=15, pady=20)

        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="#0a1530", corner_radius=12)
        self.status_frame.pack(pady=10, padx=15, fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_frame, text="Готов к работе",
            font=ctk.CTkFont(family="SF Pro Text", size=13, weight="bold"),
            text_color="#6a8a7a"
        )
        self.status_label.pack(pady=12, padx=10)

        self.progress_bar = ctk.CTkProgressBar(
            self.status_frame,
            progress_color="#4a9eff",
            fg_color="#1a2a4a",
            height=6,
            corner_radius=3
        )
        self.progress_bar.pack(pady=(0, 10), padx=15, fill="x")
        self.progress_bar.set(0)

        info_frame = ctk.CTkFrame(self.sidebar, fg_color="#0a1530", corner_radius=12)
        info_frame.pack(pady=10, padx=15, fill="both", expand=True)

        info_label = ctk.CTkLabel(
            info_frame, text="Горячие клавиши:\n\nCtrl/Cmd+O — загрузить\nCtrl/Cmd+R — распознать\nCtrl/Cmd+L — очистить\nEsc — выход",
            font=ctk.CTkFont(family="SF Pro Text", size=11),
            text_color="#5a6a8a",
            justify="left"
        )
        info_label.pack(pady=12, padx=10)

    def _build_content(self):
        header = ctk.CTkLabel(
            self.content_area, text="Распознавание текста",
            font=ctk.CTkFont(family="SF Pro Display", size=20, weight="bold"),
            text_color="#ffffff"
        )
        header.pack(anchor="w", pady=(15, 5), padx=20)

        self.image_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="#0a1020",
            corner_radius=12,
            height=280
        )
        self.image_frame.pack(fill="x", padx=15, pady=10)
        self.image_frame.pack_propagate(True)

        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Перетащите изображение\nили нажмите «Загрузить фото»",
            font=ctk.CTkFont(family="SF Pro Text", size=14),
            text_color="#4a5a7a"
        )
        self.image_label.pack(padx=10, pady=10)
        self.current_image = None

        result_header = ctk.CTkLabel(
            self.content_area, text="Результат",
            font=ctk.CTkFont(family="SF Pro Display", size=16, weight="bold"),
            text_color="#ffffff"
        )
        result_header.pack(anchor="w", pady=(10, 5), padx=20)

        self.result_box = ctk.CTkTextbox(
            self.content_area,
            font=ctk.CTkFont(family="SF Pro Text", size=13),
            corner_radius=12,
            fg_color="#0a1020",
            text_color="#d0d8e8",
            border_color="#2a3a5a",
            border_width=1,
            scrollbar_button_color="#2a3a5a"
        )
        self.result_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _set_image_label(self, pil_image=None, text=None):
        """Пересоздание image_label, чтобы избежать устаревших image-ссылок"""
        if hasattr(self, 'image_label') and self.image_label is not None:
            try:
                self.image_label.destroy()
            except Exception:
                pass

        image_text = text if text is not None else ""
        ctk_img = None
        if pil_image is not None:
            ctk_img = ctk.CTkImage(pil_image, size=pil_image.size)

        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text=image_text,
            image=ctk_img,
            font=ctk.CTkFont(family="SF Pro Text", size=14),
            text_color="#4a5a7a"
        )
        self.image_label.pack(padx=10, pady=10)

        self.current_image = ctk_img
        self.image_label._current_image = ctk_img

        # Включаем визуальные эффекты для плавности
        if pil_image:
            self._animate_frame_height(max(pil_image.size[1], 280))

    
    def _update_animation_indicator(self):
        """Обновление маленького анимированного индикатора в статусе"""
        if not self.is_showing_animation or not self.animation_indicator:
            return
        
        if self.animation_indicator.get_frames_count() == 0:
            return
        
        frame = self.animation_indicator.get_frame(self.animation_index)
        if frame is not None:
            # Конвертация PIL Image в PhotoImage
            photo = ImageTk.PhotoImage(frame)
            
            # Обновление лабели в status_frame рядом со статусом
            if not hasattr(self, 'animation_photo_label'):
                # Создание лабели если её еще нет
                self.animation_photo_label = Label(
                    self.status_frame,
                    image=None,
                    bg="#0a1530"
                )
                self.animation_photo_label.pack(side="right", padx=10, pady=12)
            
            self.animation_photo_label.config(image=photo)
            self.animation_photo_label.image = photo  # Сохранение ссылки
        
        self.animation_index += 1
        self.after(75, self._update_animation_indicator)  # ~13 FPS для плавной анимации
    
    def _start_animation_indicator(self):
        """Запуск маленького анимированного индикатора"""
        if self.animation_indicator and not self.is_showing_animation:
            self.is_showing_animation = True
            self.animation_index = 0
            self._update_animation_indicator()
    
    def _stop_animation_indicator(self):
        """Остановка маленького анимированного индикатора"""
        self.is_showing_animation = False
        if hasattr(self, 'animation_photo_label'):
            self.animation_photo_label.config(image=None)
            self.animation_photo_label.pack_forget()

    def load_image(self):
        """Загрузка изображения с проверками и обработкой ошибок"""
        if self.is_processing:
            self._set_status("Подождите, идет обработка", "#f39c12")
            logger.debug("Load image blocked: processing in progress")
            return
        
        try:
            path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp")])
            if not path:
                logger.debug("Load image cancelled by user")
                return
            
            # Проверка существования файла
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл не найден: {path}")
            
            # Проверка размера файла (максимум 50MB)
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > 50:
                raise ValueError(f"Файл слишком большой ({file_size_mb:.1f}MB). Максимум 50MB")
            
            self.image_path = path
            
            # Загрузка и валидация изображения
            img = Image.open(path)
            img.verify()
            img = Image.open(path)  # Переоткрытие после verify()
            
            # Получение и логирование информации об изображении
            logger.info(f"Loaded image: {path}, size: {img.size}, format: {img.format}")
            
            # Правильный расчет размера после thumbnail
            img.thumbnail((IMAGE_PREVIEW_WIDTH, IMAGE_PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
            display_size = img.size

            self._set_image_label(pil_image=img, text="")

            # Включаем визуальные эффекты
            self._animate_frame_height(max(display_size[1], 280))
            self._pulse_image_frame(count=4, interval=80)
            self._fade_window_to(target_alpha=1.0, duration=180, steps=6)

            # Очистка результата при загрузке нового изображения
            self.result_box.delete("1.0", "end")
            
            self._set_status("Изображение загружено", "#4a9eff")
            logger.info("Image loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            self._set_status("Файл не найден", "#e74c3c")
            self.image_path = None
            
        except ValueError as e:
            logger.error(f"Invalid image: {e}")
            self._set_status(str(e)[:60], "#e74c3c")
            self.image_path = None
            
        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
            error_msg = str(e)[:50]
            self._set_status(f"Ошибка: {error_msg}", "#e74c3c")
            self.image_path = None

    def recognize(self):
        """Распознавание текста из изображения"""
        if not self.image_path:
            self._set_status("Сначала загрузите изображение", "#e74c3c")
            logger.debug("Recognize blocked: no image path")
            return
        if self.is_processing:
            self._set_status("Распознавание уже идет", "#f39c12")
            logger.debug("Recognize blocked: already processing")
            return
        
        # Проверка, что файл все еще существует
        if not os.path.exists(self.image_path):
            self._set_status("Файл был удален", "#e74c3c")
            logger.warning(f"Image file deleted: {self.image_path}")
            self.image_path = None
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        self._set_status("Распознаю...", "#f39c12")
        self.progress_bar.set(0)
        self._animate_progress()
        
        # Запуск маленькой анимации-индикатора
        self._start_animation_indicator()
        
        logger.info(f"Started recognition for: {self.image_path}")

        def process():
            try:
                if self.ocr is None:
                    raise RuntimeError("OCR model not initialized")

                result = self.ocr.recognize(self.image_path)
                text = result.get("text", "")
                if not text:
                    text = "Текст не найден. Проверьте качество изображения и модель."

                stats = []
                if result.get("detections") is not None:
                    stats.append(f"символы: {result.get('detections')}" )
                if result.get("letters") is not None:
                    stats.append(f"буквы: {result.get('letters')}" )
                if result.get("punct") is not None:
                    stats.append(f"пунктуация: {result.get('punct')}" )

                if stats:
                    text = "[" + ", ".join(stats) + "]\n\n" + text

                logger.info("Recognition completed successfully")
                self.after(0, lambda: self._show_result(text))
            except Exception as exc:
                err_msg = f"Ошибка при распознавании: {str(exc)[:80]}\nПопробуйте еще раз"
                logger.error(f"Error during recognition: {exc}", exc_info=True)
                self.after(0, lambda msg=err_msg: self._show_result(msg))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _animate_progress(self):
        """Анимация прогресса во время обработки"""
        if not self.is_processing:
            return
        current = self.progress_bar.get()
        if current < 0.85:
            self.progress_bar.set(current + 0.15)
            self.after(300, self._animate_progress)
        else:
            # Замедляем анимацию в конце
            self.progress_bar.set(current + 0.02)
            self.after(300, self._animate_progress)

    def _show_result(self, text):
        """Отображение результата распознавания"""
        try:
            # Остановка анимации-индикатора
            self._stop_animation_indicator()
            
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", text)
            logger.debug("Result displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying result: {e}", exc_info=True)
        finally:
            self.is_processing = False
            self.btn_run.configure(state="normal")
            self.btn_load.configure(state="normal")
            self.progress_bar.set(1.0)
            self._set_status("Готов к работе", "#6a8a7a")

    def clear(self):
        """Очистка интерфейса"""
        if self.is_processing:
            self._set_status("Подождите, идет обработка", "#f39c12")
            logger.debug("Clear blocked: processing in progress")
            return
        
        # Остановка анимации если она проигрывается
        if self.is_showing_animation:
            self._stop_animation_indicator()
        
        self._set_image_label(text="Перетащите изображение\nили нажмите «Загрузить фото»")
        self._animate_frame_height(280)
        self.current_image = None
        self.result_box.delete("1.0", "end")
        self.image_path = None
        self.progress_bar.set(0)
        self._set_status("Готов к работе", "#6a8a7a")
        logger.info("Application cleared")

    def _set_status(self, text, color="#6a8a7a"):
        """Обновление статуса в интерфейсе"""
        self.status_label.configure(text=text, text_color=color)


if __name__ == "__main__":
    try:
        logger.info(f"Starting G-Vision OCR v{__version__} (macOS)")
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)
