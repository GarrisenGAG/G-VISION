from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageDraw
from tkinter import filedialog

from train import GVisionOCR

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    WIDTH = 1400
    HEIGHT = 920

    BG = "#020208"
    SIDEBAR = "#0a1a4b"
    PANEL = "#0a1a4b"
    INPUT = "#14306a"
    BTN = "#1e40af"
    BTN_HOVER = "#15307f"
    GREEN = "#22c55e"
    RED = "#ef4444"
    TEXT = "#e6eefc"
    SUBTEXT = "#94a3b8"

    def __init__(self) -> None:
        super().__init__()

        self.overrideredirect(True)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(1100, 760)
        self.title("G-Vision")
        self.configure(fg_color=self.BG)

        self._image_path: Optional[Path] = None
        self._preview = None
        self._ocr = None
        self._drag_position: Optional[Tuple[int, int]] = None
        self._processing_animation_active = False
        self._processing_dots = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        # параметры инференса
        self._model_path = Path(__file__).parent / "best.pt"
        self._device = "auto"
        self._use_compile = False

        self._settings_window = None

        self._build()
        self._init_ocr()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_header()

        self.sidebar = ctk.CTkFrame(
            self,
            width=360,
            fg_color=self.SIDEBAR,
            corner_radius=32,
        )
        self.sidebar.grid(row=1, column=0, sticky="ns", padx=16, pady=16)

        self.content = ctk.CTkFrame(self, fg_color=self.BG, corner_radius=32)
        self.content.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=16,
            pady=16,
        )
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_preview()
        self._build_result()
        self._build_settings_button()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(
            self,
            fg_color=self.SIDEBAR,
            height=124,
            corner_radius=28,
        )
        header.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=16,
            pady=(12, 12),
        )
        header.grid_columnconfigure(0, weight=0)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, weight=0)

        logo_path = Path(__file__).parent / "logo.png"
        left_frame = ctk.CTkFrame(header, fg_color=self.SIDEBAR, corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="w", padx=(18, 8), pady=12)
        left_frame.grid_columnconfigure(0, weight=0)
        left_frame.grid_columnconfigure(1, weight=0)

        if logo_path.exists():
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                try:
                    datas = list(logo_img.getdata())
                    new_data = []
                    for item in datas:
                        r, g, b, a = item
                        if r >= 240 and g >= 240 and b >= 240:
                            new_data.append((255, 255, 255, 0))
                        else:
                            new_data.append((r, g, b, a))
                    logo_img.putdata(new_data)
                except Exception:
                    pass
                logo_img.thumbnail((48, 48))
                self._logo_image = ctk.CTkImage(logo_img, size=(48, 48))

                logo_label = ctk.CTkLabel(left_frame, image=self._logo_image, text="")
                logo_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
                logo_label.bind("<ButtonPress-1>", self.start_move)
            except Exception:
                pass
        else:
            logo_bg = ctk.CTkFrame(
                left_frame,
                width=48,
                height=48,
                fg_color=self.BTN,
                corner_radius=28,
            )
            logo_bg.grid(row=0, column=0, sticky="w", padx=(0, 8))
            logo_bg.grid_propagate(False)

            logo_label = ctk.CTkLabel(
                logo_bg,
                text="G",
                font=("Arial", 18, "bold"),
                text_color="#ffffff",
            )
            logo_label.place(relx=0.5, rely=0.5, anchor="center")
            logo_bg.bind("<ButtonPress-1>", self.start_move)
            logo_label.bind("<ButtonPress-1>", self.start_move)

        title = ctk.CTkLabel(
            left_frame,
            text="G-Vision",
            font=("Arial", 18, "bold"),
            text_color=self.TEXT,
        )
        title.grid(row=0, column=1, sticky="w")
        title.bind("<ButtonPress-1>", self.start_move)
        title.bind("<B1-Motion>", self.move_window)

        progress_container = ctk.CTkFrame(
            header,
            fg_color=self.BTN,
            corner_radius=26,
        )
        progress_container.grid(row=0, column=1, sticky="ew", padx=8, pady=12)
        progress_container.grid_propagate(False)
        progress_container.configure(height=42)

        self.header_status_label = ctk.CTkLabel(
            progress_container,
            text="Готов",
            text_color=self.SUBTEXT,
            font=("Arial", 13, "bold"),
        )
        self.header_status_label.place(relx=0.5, rely=0.5, anchor="center")

        self.close_button = ctk.CTkButton(
            header,
            text="✕",
            width=44,
            height=40,
            corner_radius=24,
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            command=self.close,
            font=("Arial", 14, "bold"),
        )
        self.close_button.grid(row=0, column=2, sticky="e", padx=16, pady=12)

    def _build_sidebar(self) -> None:
        self.btn_load = self.make_btn("Загрузить изображение", self.load_image)
        self.btn_load.pack(fill="x", padx=20, pady=(20, 8))

        self.btn_run = self.make_btn("Распознать текст", self.recognize)
        self.btn_run.pack(fill="x", padx=20, pady=8)

        self.btn_clear = self.make_btn("Очистить", self.clear_all, red=True)
        self.btn_clear.pack(fill="x", padx=20, pady=(8, 20))

    def _build_settings_button(self) -> None:
        self.settings_button = ctk.CTkButton(
            self.sidebar,
            text="⚙",
            width=52,
            height=52,
            corner_radius=26,
            fg_color=self.INPUT,
            hover_color=self.BTN_HOVER,
            command=self.open_settings,
            font=("Arial", 18, "bold"),
        )
        self.settings_button.pack(side="bottom", anchor="w", padx=20, pady=20)

    def _build_preview(self) -> None:
        self.preview_card = ctk.CTkFrame(
            self.content,
            fg_color=self.PANEL,
            corner_radius=32,
        )
        self.preview_card.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.preview_card,
            text="Изображение",
            font=("Arial", 24, "bold"),
            text_color=self.TEXT,
        )
        title.pack(anchor="w", padx=24, pady=20)

        self.preview_container = ctk.CTkFrame(
            self.preview_card,
            fg_color=self.INPUT,
            corner_radius=32,
        )
        self.preview_container.pack(expand=True, fill="both", padx=20, pady=(0, 24))
        self.preview_container.grid_rowconfigure(0, weight=1)
        self.preview_container.grid_columnconfigure(0, weight=1)

        self.preview_box = ctk.CTkFrame(
            self.preview_container,
            fg_color=self.INPUT,
            corner_radius=32,
        )
        self.preview_box.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.preview_box.grid_rowconfigure(0, weight=1)
        self.preview_box.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            self.preview_box,
            text="",
            fg_color=self.INPUT,
            corner_radius=0,
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _build_result(self) -> None:
        self.result_card = ctk.CTkFrame(
            self.content,
            fg_color=self.PANEL,
            corner_radius=32,
        )
        self.result_card.grid(row=2, column=0, sticky="nsew")
        self.result_card.grid_rowconfigure(1, weight=1)
        self.result_card.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.result_card,
            text="Распознанный текст",
            font=("Arial", 24, "bold"),
            text_color=self.TEXT,
        )
        title.pack(anchor="w", padx=24, pady=20)

        self.result = ctk.CTkTextbox(
            self.result_card,
            fg_color=self.INPUT,
            font=("Arial", 16),
            corner_radius=28,
        )
        self.result.pack(expand=True, fill="both", padx=20)

        self.copy_btn = ctk.CTkButton(
            self.result_card,
            text="Копировать текст",
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            command=self.copy_text,
            width=220,
            height=48,
            corner_radius=28,
        )
        self.copy_btn.pack(anchor="e", padx=20, pady=18)

    def make_btn(self, text: str, cmd, red: bool = False) -> ctk.CTkButton:
        if red:
            return ctk.CTkButton(
                self.sidebar,
                text=text,
                height=56,
                font=("Arial", 16, "bold"),
                fg_color="#ffffff",
                text_color=self.BTN,
                hover_color="#dee8ff",
                border_width=2,
                border_color=self.BTN,
                command=cmd,
                corner_radius=28,
            )

        return ctk.CTkButton(
            self.sidebar,
            text=text,
            height=56,
            font=("Arial", 16, "bold"),
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            command=cmd,
            corner_radius=28,
        )

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        self._image_path = Path(file_path)
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((900, 650))
        radius = min(32, min(image.size) // 8)
        rounded = self._make_rounded_image(image, radius=radius)

        self._preview = ctk.CTkImage(rounded, size=rounded.size)
        self.preview_label.configure(image=self._preview, text="")

        self.update_status("Изображение загружено", self.TEXT)

    def _init_ocr(self) -> None:
        if not self._model_path.exists():
            self.update_status(
                f"Модель не найдена: {self._model_path.name} — укажите путь в настройках",
                self.RED,
            )
            self._ocr = None
            return

        try:
            self._ocr = GVisionOCR(
                str(self._model_path),
                device=self._device,
                use_compile=self._use_compile,
            )
            self.update_status(f"Модель загружена: {self._model_path.name}", self.TEXT)
        except Exception as error:
            self._ocr = None
            self.update_status(f"Ошибка загрузки OCR: {error}", self.RED)

    def open_settings(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.lift()
            return

        self._settings_window = ctk.CTkToplevel(self)
        self._settings_window.title("Настройки")
        self._settings_window.geometry("560x380")
        self._settings_window.configure(fg_color=self.BG)
        self._settings_window.resizable(False, False)

        header = ctk.CTkLabel(
            self._settings_window,
            text="Настройки модели",
            font=("Arial", 20, "bold"),
            text_color=self.TEXT,
        )
        header.pack(anchor="w", padx=24, pady=(24, 8))

        body = ctk.CTkFrame(
            self._settings_window,
            fg_color=self.PANEL,
            corner_radius=28,
        )
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        body.grid_columnconfigure(0, weight=1)

        self._build_settings_panel(body)

    def _build_settings_panel(self, parent: ctk.CTkFrame) -> None:
        # блок: путь к модели
        model_block = ctk.CTkFrame(parent, fg_color=self.INPUT, corner_radius=24)
        model_block.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        model_block.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            model_block,
            text="Путь к модели (.pt / .pth)",
            font=("Arial", 14, "bold"),
            text_color=self.TEXT,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self.model_path_var = ctk.StringVar(value=str(self._model_path))
        ctk.CTkEntry(
            model_block,
            textvariable=self.model_path_var,
            fg_color=self.BG,
            corner_radius=16,
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))

        ctk.CTkButton(
            model_block,
            text="Выбрать файл",
            width=160,
            height=42,
            corner_radius=20,
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            command=self.choose_model_file,
        ).grid(row=2, column=0, sticky="e", padx=16, pady=(0, 16))

        # блок: устройство и компиляция
        infer_block = ctk.CTkFrame(parent, fg_color=self.INPUT, corner_radius=24)
        infer_block.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        infer_block.grid_columnconfigure(1, weight=1)

        self.device_var = ctk.StringVar(value=self._device)
        ctk.CTkLabel(
            infer_block,
            text="Устройство",
            font=("Arial", 14, "bold"),
            text_color=self.TEXT,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))
        ctk.CTkOptionMenu(
            infer_block,
            values=["auto", "cpu", "cuda"],
            variable=self.device_var,
            width=180,
            corner_radius=20,
        ).grid(row=0, column=1, sticky="e", padx=16, pady=(16, 8))

        self.use_compile_var = ctk.BooleanVar(value=self._use_compile)
        ctk.CTkCheckBox(
            infer_block,
            text="Компиляция модели (torch.compile)",
            variable=self.use_compile_var,
            corner_radius=16,
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 16))

        # кнопка сохранить
        ctk.CTkButton(
            parent,
            text="Сохранить и перезагрузить модель",
            width=260,
            height=48,
            corner_radius=24,
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            command=self.apply_settings,
        ).grid(row=2, column=0, sticky="e", padx=16, pady=(8, 16))

    def choose_model_file(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch модели", "*.pt *.pth"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)

    def apply_settings(self) -> None:
        try:
            self._model_path = Path(self.model_path_var.get())
            self._device = self.device_var.get()
            self._use_compile = self.use_compile_var.get()
            self._init_ocr()
        except Exception as error:
            self.update_status(f"Ошибка настроек: {error}", self.RED)

    def recognize(self) -> None:
        if not self._ocr:
            self.update_status("OCR не загружен — проверьте настройки", self.RED)
            return
        if not self._image_path:
            self.update_status("Сначала загрузите изображение", self.RED)
            return

        self._processing_animation_active = True
        self._processing_dots = 0
        self.update_status("Разглядываю изображение...", self.TEXT)
        self._animate_processing()
        self.executor.submit(self._worker)

    def _worker(self) -> None:
        try:
            result = self._ocr.recognize(str(self._image_path))
            text = result.get("text", "")
            self.after(0, lambda: self._show_result(text))
        except Exception as error:
            self.after(0, lambda: self.update_status(f"Ошибка распознавания: {error}", self.RED))
        finally:
            self.after(0, self._finish_processing)

    def _animate_processing(self) -> None:
        if not self._processing_animation_active:
            return
        phrases = [
            "Разглядываю изображение...",
            "Ищу буквы...",
            "Перевожу в текст...",
            "Формирую результат...",
        ]
        self._processing_dots = (self._processing_dots + 1) % len(phrases)
        self.header_status_label.configure(text=phrases[self._processing_dots])
        self.after(700, self._animate_processing)

    def _finish_processing(self) -> None:
        self._processing_animation_active = False
        self.update_status("Распознавание завершено", self.GREEN)

    def _show_result(self, text: str) -> None:
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.update_status("Текст успешно распознан", self.GREEN)

    def copy_text(self) -> None:
        text = self.result.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_status("Текст скопирован в буфер", self.GREEN)

    def clear_all(self) -> None:
        self.preview_label.configure(image=None)
        self._preview = None
        self._image_path = None
        self.result.delete("1.0", "end")
        self.update_status("Готов — загрузите новое изображение", self.GREEN)

    def _make_rounded_image(self, image: Image.Image, radius: int) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, image.width, image.height), radius=radius, fill=255)
        image.putalpha(mask)

        rounded = Image.new("RGBA", image.size, self.INPUT)
        rounded.paste(image, (0, 0), image)
        return rounded

    def update_status(self, text: str, text_color: Optional[str] = None) -> None:
        try:
            self.header_status_label.configure(
                text=text,
                text_color=text_color or self.SUBTEXT,
            )
        except Exception:
            pass

    def start_move(self, event) -> None:
        self._drag_position = (event.x_root, event.y_root)

    def move_window(self, event) -> None:
        if self._drag_position is None:
            return
        delta_x = event.x_root - self._drag_position[0]
        delta_y = event.y_root - self._drag_position[1]
        x = self.winfo_x() + delta_x
        y = self.winfo_y() + delta_y
        self.geometry(f"+{x}+{y}")
        self._drag_position = (event.x_root, event.y_root)

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
