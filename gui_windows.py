"""G-Vision OCR - Windows Version"""
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

# Constants
__version__ = "1.0.0"
APP_NAME = "G-Vision"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 740  # +40 for title bar
MIN_WIDTH = 700
MIN_HEIGHT = 540
IMAGE_PREVIEW_WIDTH = 600
IMAGE_PREVIEW_HEIGHT = 400

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnimationIndicator:
    """Small animated loading indicator with transparency"""
    def __init__(self, video_path, size=48):
        self.video_path = video_path
        self.size = size
        self.frames = []
        self.current_frame_index = 0
        self._load_frames()
    
    def _load_frames(self):
        """Load and scale frames"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {self.video_path}")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame, (self.size, self.size))
                pil_image = Image.fromarray(frame_resized).convert("RGBA")

                arr = np.array(pil_image)
                black_mask = (arr[:, :, 0] < 30) & (arr[:, :, 1] < 30) & (arr[:, :, 2] < 30)
                arr[black_mask, 3] = 0
                arr[~black_mask, 3] = 200
                pil_image = Image.fromarray(arr, mode='RGBA')

                self.frames.append(pil_image)
            
            cap.release()
            logger.info(f"Loaded {len(self.frames)} indicator frames from {self.video_path}")
        except Exception as e:
            logger.error(f"Error loading frames: {e}")
    
    def get_frame(self, index):
        """Get frame by index"""
        if len(self.frames) == 0:
            return None
        index = index % len(self.frames)
        return self.frames[index]
    
    def get_frames_count(self):
        """Get frames count"""
        return len(self.frames)
    
    def reset_index(self):
        """Reset index"""
        self.current_frame_index = 0


class App(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            self.title(APP_NAME)
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
            self.minsize(MIN_WIDTH, MIN_HEIGHT)

            # Center window on screen
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - WINDOW_WIDTH) // 2
            y = (screen_h - WINDOW_HEIGHT) // 2
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

            self.configure(fg_color="#010101")
            self.overrideredirect(True)  # Custom frame instead of standard

            # Variables for window movement
            self._drag_data = {"x": 0, "y": 0}

            # Animation initialization
            self.animation_indicator = None
            self.animation_index = 0
            self.is_showing_animation = False
            self._init_animation_indicator()

            # Custom title bar
            self._build_title_bar()

            self.main_container = ctk.CTkFrame(self, fg_color="transparent")
            self.main_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

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

            # Initialize state
            self.image_path = None
            self.is_processing = False
            self.current_image = None
            self.ocr = None

            self._build_sidebar()
            self._build_content()
            self._setup_shortcuts()
            self._init_ocr()

            logger.info("App initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing app: {e}", exc_info=True)
            raise

    def _build_title_bar(self):
        """Create custom title bar"""
        title_bar = ctk.CTkFrame(self, fg_color="#0a0a0a", height=40)
        title_bar.pack(side="top", fill="x", padx=0, pady=0)
        title_bar.pack_propagate(False)

        # Left part - app name
        app_label = ctk.CTkLabel(
            title_bar, text="G-Vision OCR",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color="#ffffff"
        )
        app_label.pack(side="left", padx=15, pady=8)

        # Right part - buttons
        button_frame = ctk.CTkFrame(title_bar, fg_color="transparent")
        button_frame.pack(side="right", padx=10, pady=5)

        # Minimize button
        minimize_btn = ctk.CTkButton(
            button_frame,
            text="−",
            width=30,
            height=30,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            text_color="#ffffff",
            command=self._minimize_window
        )
        minimize_btn.pack(side="left", padx=3)

        # Close button
        close_btn = ctk.CTkButton(
            button_frame,
            text="✕",
            width=30,
            height=30,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
            text_color="#ffffff",
            command=self.destroy
        )
        close_btn.pack(side="left", padx=3)

        # Bind window dragging
        title_bar.bind("<Button-1>", self._on_title_press)
        title_bar.bind("<B1-Motion>", self._on_title_drag)
        app_label.bind("<Button-1>", self._on_title_press)
        app_label.bind("<B1-Motion>", self._on_title_drag)

    def _on_title_press(self, event):
        """Start moving window"""
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root

    def _on_title_drag(self, event):
        """Move window while dragging"""
        dx = event.x_root - self._drag_data["x"]
        dy = event.y_root - self._drag_data["y"]
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        self.geometry(f"+{x}+{y}")
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root

    def _minimize_window(self):
        """Minimize window"""
        self.overrideredirect(False)
        self.state("iconic")
        self.overrideredirect(True)

    def _setup_shortcuts(self):
        """Setup hotkeys"""
        shortcuts = [
            ('<Escape>', lambda e: self.destroy()),
            ('<Control-o>', lambda e: self.load_image()),
            ('<Control-r>', lambda e: self.recognize()),
            ('<Control-l>', lambda e: self.clear()),
        ]
        for key, command in shortcuts:
            self.bind(key, command)

    def _build_sidebar(self):
        title = ctk.CTkLabel(
            self.sidebar, text="Control",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color="#ffffff"
        )
        title.pack(pady=(20, 25), padx=15)

        btn_style = {
            "font": ctk.CTkFont(family="Arial", size=14, weight="bold"),
            "height": 44,
            "corner_radius": 10,
            "fg_color": "#1a2a5a",
            "hover_color": "#243a7a",
            "text_color": "#ffffff",
        }

        self.btn_load = ctk.CTkButton(self.sidebar, text="📁 Load photo", command=self.load_image, **btn_style)
        self.btn_load.pack(pady=6, padx=15, fill="x")

        self.btn_run = ctk.CTkButton(self.sidebar, text="🔍 Recognize", command=self.recognize, **btn_style)
        self.btn_run.pack(pady=6, padx=15, fill="x")

        self.btn_clear = ctk.CTkButton(self.sidebar, text="🗑️ Clear", command=self.clear, **btn_style)
        self.btn_clear.pack(pady=6, padx=15, fill="x")

        separator = ctk.CTkFrame(self.sidebar, height=1, fg_color="#2a3a6a")
        separator.pack(fill="x", padx=15, pady=20)

        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="#0a1530", corner_radius=12)
        self.status_frame.pack(pady=10, padx=15, fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_frame, text="Ready to work",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
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
            info_frame, text="Hotkeys:\n\nCtrl+O — load\nCtrl+R — recognize\nCtrl+L — clear\nEsc — exit",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color="#5a6a8a",
            justify="left"
        )
        info_label.pack(pady=12, padx=10)

    def _build_content(self):
        header = ctk.CTkLabel(
            self.content_area, text="Text Recognition",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
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
            text="Drag and drop image\nor click 'Load photo'",
            font=ctk.CTkFont(family="Arial", size=14),
            text_color="#4a5a7a"
        )
        self.image_label.pack(padx=10, pady=10)
        self.current_image = None

        result_header = ctk.CTkLabel(
            self.content_area, text="Result",
            font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
            text_color="#ffffff"
        )
        result_header.pack(anchor="w", pady=(10, 5), padx=20)

        self.result_box = ctk.CTkTextbox(
            self.content_area,
            font=ctk.CTkFont(family="Arial", size=13),
            corner_radius=12,
            fg_color="#0a1020",
            text_color="#d0d8e8",
            border_color="#2a3a5a",
            border_width=1,
            scrollbar_button_color="#2a3a5a"
        )
        self.result_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _set_image_label(self, pil_image=None, text=None):
        """Recreate image_label to avoid stale image references"""
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
            font=ctk.CTkFont(family="Arial", size=14),
            text_color="#4a5a7a"
        )
        self.image_label.pack(padx=10, pady=10)

        self.current_image = ctk_img
        self.image_label._current_image = ctk_img

    def _init_ocr(self):
        """Initialize OCR model from best.pt or g-vision-config.json."""
        try:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(root_dir, "best.pt")
            config_path = os.path.join(root_dir, "g-vision-config.json")
            config_model_path = None

            if os.path.exists(config_path):
                try:
                    with open(config_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    model_from_cfg = cfg.get("model")
                    if model_from_cfg:
                        candidate = model_from_cfg if os.path.isabs(model_from_cfg) else os.path.join(root_dir, model_from_cfg)
                        if os.path.exists(candidate):
                            config_model_path = candidate
                except Exception as ex:
                    logger.warning(f"Failed to load OCR config: {ex}")

            if config_model_path and os.path.exists(config_model_path):
                model_path = config_model_path

            if not os.path.exists(model_path):
                fallback_paths = [
                    os.path.join(root_dir, "best_model.pt"),
                    os.path.join(root_dir, "G-Vision.pt"),
                    os.path.join(root_dir, "G-VISION.pt"),
                    os.path.join(root_dir, "best.pt"),
                    os.path.join(root_dir, "runs", "ocr_final", "best_model.pt")
                ]
                for p in fallback_paths:
                    if os.path.exists(p):
                        model_path = p
                        break

            if not os.path.exists(model_path):
                self._set_status("OCR model not found", "#e74c3c")
                logger.warning(f"OCR model not found: {model_path}")
                return

            config_arg = config_path if os.path.exists(config_path) else None
            from train import GVisionOCR
            self.ocr = GVisionOCR(model_path, config_arg)
            self._set_status("OCR model loaded", "#2ecc71")
            logger.info(f"OCR model initialized from {model_path}, config {config_arg}")
        except Exception as e:
            logger.error(f"Error initializing OCR model: {e}", exc_info=True)
            self.ocr = None
            self._set_status("OCR loading error", "#e74c3c")

    def _init_animation_indicator(self):
        """Initialize small animated indicator"""
        try:
            video_path = os.path.join(os.path.dirname(__file__), "wait_anim.mkv")
            if os.path.exists(video_path):
                self.animation_indicator = AnimationIndicator(video_path, size=32)
                logger.info(f"Animation indicator initialized with {self.animation_indicator.get_frames_count()} frames")
            else:
                logger.warning(f"Animation file not found: {video_path}")
        except Exception as e:
            logger.error(f"Error initializing animation indicator: {e}")

    def _update_animation_indicator(self):
        """Update small animated indicator in status"""
        if not self.is_showing_animation or not self.animation_indicator:
            return
        
        if self.animation_indicator.get_frames_count() == 0:
            return
        
        frame = self.animation_indicator.get_frame(self.animation_index)
        if frame is not None:
            photo = ImageTk.PhotoImage(frame)
            
            if not hasattr(self, 'animation_photo_label'):
                self.animation_photo_label = Label(
                    self.status_frame,
                    image=None,
                    bg="#0a1530"
                )
                self.animation_photo_label.pack(side="right", padx=10, pady=12)
            
            self.animation_photo_label.config(image=photo)
            self.animation_photo_label.image = photo
        
        self.animation_index += 1
        self.after(75, self._update_animation_indicator)
    
    def _start_animation_indicator(self):
        """Start small animated indicator"""
        if self.animation_indicator and not self.is_showing_animation:
            self.is_showing_animation = True
            self.animation_index = 0
            self._update_animation_indicator()
    
    def _stop_animation_indicator(self):
        """Stop small animated indicator"""
        self.is_showing_animation = False
        if hasattr(self, 'animation_photo_label'):
            self.animation_photo_label.config(image=None)
            self.animation_photo_label.pack_forget()

    def load_image(self):
        """Load image with validation and error handling"""
        if self.is_processing:
            self._set_status("Wait, processing in progress", "#f39c12")
            logger.debug("Load image blocked: processing in progress")
            return
        
        try:
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
            if not path:
                logger.debug("Load image cancelled by user")
                return
            
            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            # Check file size (max 50MB)
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > 50:
                raise ValueError(f"File too large ({file_size_mb:.1f}MB). Max 50MB")
            
            self.image_path = path
            
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
            
            logger.info(f"Loaded image: {path}, size: {img.size}, format: {img.format}")
            
            img.thumbnail((IMAGE_PREVIEW_WIDTH, IMAGE_PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
            display_size = img.size

            self._set_image_label(pil_image=img, text="")
            self.result_box.delete("1.0", "end")
            
            self._set_status("Image loaded", "#4a9eff")
            logger.info("Image loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            self._set_status("File not found", "#e74c3c")
            self.image_path = None
            
        except ValueError as e:
            logger.error(f"Invalid image: {e}")
            self._set_status(str(e)[:60], "#e74c3c")
            self.image_path = None
            
        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
            error_msg = str(e)[:50]
            self._set_status(f"Error: {error_msg}", "#e74c3c")
            self.image_path = None

    def recognize(self):
        """Recognize text from image"""
        if not self.image_path:
            self._set_status("Load image first", "#e74c3c")
            logger.debug("Recognize blocked: no image path")
            return
        if self.is_processing:
            self._set_status("Recognition already running", "#f39c12")
            logger.debug("Recognize blocked: already processing")
            return
        
        if not os.path.exists(self.image_path):
            self._set_status("File was deleted", "#e74c3c")
            logger.warning(f"Image file deleted: {self.image_path}")
            self.image_path = None
            return

        self.is_processing = True
        self.btn_run.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        self._set_status("Recognizing...", "#f39c12")
        self.progress_bar.set(0)
        
        self._start_animation_indicator()
        
        logger.info(f"Started recognition for: {self.image_path}")

        def process():
            try:
                if self.ocr is None:
                    raise RuntimeError("OCR model not initialized")

                result = self.ocr.recognize(self.image_path)
                text = result.get("text", "")
                if not text:
                    text = "No text found. Check the image quality and model."

                logger.info("Recognition completed successfully")
                self.after(0, lambda: self._show_result(text))
            except Exception as e:
                logger.error(f"Error during recognition: {e}", exc_info=True)
                self.after(0, lambda: self._show_result(f"Recognition error: {str(e)[:80]}. Try again."))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _show_result(self, text):
        """Display recognition result"""
        try:
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
            self._set_status("Ready to work", "#6a8a7a")

    def clear(self):
        """Clear interface"""
        if self.is_processing:
            self._set_status("Wait, processing in progress", "#f39c12")
            logger.debug("Clear blocked: processing in progress")
            return
        
        if self.is_showing_animation:
            self._stop_animation_indicator()
        
        self._set_image_label(text="Drag and drop image\nor click 'Load photo'")
        self.current_image = None
        self.result_box.delete("1.0", "end")
        self.image_path = None
        self.progress_bar.set(0)
        self._set_status("Ready to work", "#6a8a7a")
        logger.info("Application cleared")

    def _set_status(self, text, color="#6a8a7a"):
        """Update status in the interface"""
        self.status_label.configure(text=text, text_color=color)


if __name__ == "__main__":
    try:
        logger.info(f"Starting G-Vision OCR v{__version__} (Windows)")
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)
