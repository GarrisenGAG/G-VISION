import sys
import math
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RoundScrollbar(tk.Canvas):
    def __init__(self, parent, orient="vertical", command=None,
                 bg="#14306a", thumb_color="#1e40af", hover_color="#2a50cf", **kwargs):
        super().__init__(parent, bg=bg, highlightthickness=0, borderwidth=0, **kwargs)
        self._orient = orient
        self._command = command
        self._thumb_color = thumb_color
        self._hover_color = hover_color
        self._bg = bg
        self._pos = (0.0, 1.0)
        self._dragging = False
        self._drag_start = None

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self._draw(hover=True))
        self.bind("<Leave>", lambda e: self._draw(hover=False))
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, lo, hi):
        self._pos = (float(lo), float(hi))
        self._draw()

    def _draw(self, hover=False):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        lo, hi = self._pos
        color = self._hover_color if hover else self._thumb_color

        if self._orient == "vertical":
            pad = 2
            x0, x1 = pad, w - pad
            y0 = lo * h + pad
            y1 = hi * h - pad
        else:
            pad = 2
            y0, y1 = pad, h - pad
            x0 = lo * w + pad
            x1 = hi * w - pad

        r = (x1 - x0) // 2 if self._orient == "vertical" else (y1 - y0) // 2
        r = max(r, 4)
        self.create_rounded_rect(x0, y0, x1, y1, r, fill=color)

    def create_rounded_rect(self, x0, y0, x1, y1, r, **kwargs):
        self.create_polygon(
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
            smooth=True, **kwargs
        )

    def _on_press(self, event):
        self._dragging = True
        self._drag_start = event.y if self._orient == "vertical" else event.x

    def _on_drag(self, event):
        if not self._dragging or self._drag_start is None:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        size = h if self._orient == "vertical" else w
        delta = ((event.y if self._orient == "vertical" else event.x) - self._drag_start) / size
        self._drag_start = event.y if self._orient == "vertical" else event.x
        if self._command:
            self._command("moveto", self._pos[0] + delta)

    def _on_release(self, event):
        self._dragging = False


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
    CORNER_RADIUS = 32
    TRANSPARENT = "#010101"

    def __init__(self) -> None:
        super().__init__()

        self._window_rounded = False
        self.overrideredirect(True)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(1100, 760)
        self.title("G-Vision")

        if platform.system() == "Windows":
            self.configure(fg_color=self.TRANSPARENT)
        else:
            self.configure(fg_color=self.BG)

        self._apply_window_rounding()

        self._image_path: Optional[Path] = None
        self._preview = None
        self._ocr = None
        self._drag_position: Optional[Tuple[int, int]] = None
        self._processing_animation_active = False
        self._processing_dots = 0
        self._glow_phase = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self._model_path = Path(__file__).parent / "best.pt"
        self._device = "auto"
        self._use_compile = False
        self._canvas_image_ref = None
        self._history_items = []

        self._build()
        self._init_ocr()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _apply_window_rounding(self) -> None:
        system = platform.system()
        if system == "Windows":
            # цветовой ключ: пиксели TRANSPARENT становятся прозрачными
            try:
                self.wm_attributes("-transparentcolor", self.TRANSPARENT)
                self._window_rounded = True
            except Exception:
                pass
        elif system == "Darwin":
            # alpha-прозрачность: области с alpha=0 показывают рабочий стол
            try:
                self.wm_attributes("-transparent", True)
                self._window_rounded = True
            except Exception:
                pass

    def _build(self) -> None:
        system = platform.system()
        if system == "Windows":
            # углы _bg_frame рисуются цветом TRANSPARENT (#010101) → прозрачные через -transparentcolor
            self._bg_frame = ctk.CTkFrame(
                self, fg_color=self.BG, corner_radius=self.CORNER_RADIUS,
                border_width=2, border_color="#081327",
            )
            self._bg_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self._bg_frame.lower()
        elif system == "Darwin":
            # bg_color="systemTransparent" → углы имеют alpha=0 → прозрачные через -transparent True
            self._bg_frame = ctk.CTkFrame(
                self, fg_color=self.BG, corner_radius=self.CORNER_RADIUS,
                border_width=2, border_color="#081327",
                bg_color="systemTransparent",
            )
            self._bg_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self._bg_frame.lower()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_header()

        self.sidebar = ctk.CTkFrame(
            self,
            width=360,
            fg_color=self.SIDEBAR,
            corner_radius=self.CORNER_RADIUS,
            border_width=1,
            border_color="#1a3a7a",
            bg_color=self.BG,
        )
        self.sidebar.grid(row=1, column=0, sticky="ns", padx=(16, 8), pady=(0, 16))

        self.content = ctk.CTkFrame(
            self,
            fg_color=self.BG,
            corner_radius=self.CORNER_RADIUS,
            bg_color=self.BG,
            #border_width=1,
            #border_color="#1a3a7a",
        )
        self.content.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(8, 16),
            pady=(0, 16),
        )
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_preview()
        self._build_result()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(
            self,
            fg_color=self.SIDEBAR,
            height=124,
            corner_radius=self.CORNER_RADIUS,
            border_width=1,
            border_color="#1a3a7a",
            bg_color=self.BG,
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
                scale = 8
                circle_size = 56 * scale

                logo_img = logo_img.resize((40 * scale, 40 * scale), Image.LANCZOS)

                circle = Image.new("RGBA", (circle_size, circle_size), (0, 0, 0, 0))
                mask = Image.new("L", (circle_size, circle_size), 0)
                ImageDraw.Draw(mask).ellipse((0, 0, circle_size, circle_size), fill=255)
                white = Image.new("RGBA", (circle_size, circle_size), (255, 255, 255, 255))
                circle.paste(white, mask=mask)

                offset = ((circle_size - logo_img.width) // 2, (circle_size - logo_img.height) // 2)
                circle.paste(logo_img, offset, logo_img)
                circle = circle.resize((56, 56), Image.LANCZOS)

                self._logo_image = ctk.CTkImage(circle, size=(56, 56))
                logo_label = ctk.CTkLabel(left_frame, image=self._logo_image, text="", fg_color="transparent")
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
                font=("Arial Bold", 18),
                text_color="#ffffff",
            )
            logo_label.place(relx=0.5, rely=0.5, anchor="center")
            logo_bg.bind("<ButtonPress-1>", self.start_move)
            logo_label.bind("<ButtonPress-1>", self.start_move)

        title = ctk.CTkLabel(
            left_frame,
            text="G-Vision",
            font=("Arial Bold", 24),
            text_color=self.TEXT,
        )
        title.grid(row=0, column=1, sticky="w")
        title.bind("<ButtonPress-1>", self.start_move)
        title.bind("<B1-Motion>", self.move_window)

        self.progress_container = ctk.CTkFrame(
            header,
            fg_color=self.BTN,
            corner_radius=26,
            border_width=1,
            border_color="#2a52c0",
        )
        progress_container = self.progress_container
        progress_container.grid(row=0, column=1, sticky="ew", padx=8, pady=12)
        progress_container.grid_propagate(False)
        progress_container.configure(height=42)

        self.header_status_label = ctk.CTkLabel(
            progress_container,
            text="Готов",
            text_color=self.SUBTEXT,
            font=("Arial Bold", 13),
        )
        self.header_status_label.place(relx=0.5, rely=0.5, anchor="center")
        self.header_status_label.bind("<Button-1>", self._on_model_status_click)

        close_img = Image.open(Path(__file__).parent / "x.png").convert("RGBA")
        self._close_image = ctk.CTkImage(close_img, size=(14, 14))
        self.close_button = ctk.CTkButton(
            header,
            image=self._close_image,
            text="",
            width=44,
            height=44,
            corner_radius=22,
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            border_width=1,
            border_color="#2a52c0",
            command=self.close,
        )
        self.close_button.grid(row=0, column=2, sticky="e", padx=16, pady=12)

    def _build_sidebar(self) -> None:
        self.btn_load = self.make_btn("Загрузить изображение", self.load_image)
        self.btn_load.pack(fill="x", padx=20, pady=(20, 8))

        self.btn_run = self.make_btn("Распознать текст", self.recognize)
        self.btn_run.pack(fill="x", padx=20, pady=8)

        self.btn_clear = self.make_btn("Очистить", self.clear_all, red=True)
        self.btn_clear.pack(fill="x", padx=20, pady=(8, 20))

        history_title = ctk.CTkLabel(
            self.sidebar,
            text="История",
            font=("Arial Bold", 16),
            text_color=self.SUBTEXT,
        )
        history_title.pack(anchor="w", padx=24, pady=(0, 8))

        history_outer = ctk.CTkFrame(
            self.sidebar,
            fg_color=self.INPUT,
            corner_radius=24,
            border_width=1,
            border_color="#1a3a7a",
        )
        history_outer.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._history_canvas = tk.Canvas(
            history_outer,
            bg=self.INPUT,
            highlightthickness=0,
            borderwidth=0,
        )
        self._history_canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)

        history_scroll = RoundScrollbar(
            history_outer,
            orient="vertical",
            command=self._history_canvas.yview,
            width=8,
        )
        history_scroll.pack(side="right", fill="y", pady=8, padx=(0, 6))
        self._history_canvas.configure(yscrollcommand=history_scroll.set)

        self._history_frame = tk.Frame(self._history_canvas, bg=self.INPUT)
        self._history_canvas_window = self._history_canvas.create_window(
            (0, 0), window=self._history_frame, anchor="nw"
        )

        self._history_frame.bind("<Configure>", self._on_history_resize)
        self._history_canvas.bind("<Configure>", self._on_history_canvas_resize)
        self._history_canvas.bind("<MouseWheel>", lambda e: self._history_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    def _draw_canvas_corners(self, canvas, bg_color, radius=24):
        def redraw(event=None):
            canvas.delete("corners")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 2 or h < 2:
                return
            r = radius

            for x, y, ax, ay in [
                (0, 0, 0, 0),
                (w, 0, 1, 0),
                (0, h, 0, 1),
                (w, h, 1, 1),
            ]:
                canvas.create_arc(
                    x - r * (1 - 2*ax), y - r * (1 - 2*ay),
                    x + r * (2*ax - 0) if ax == 0 else x - r,
                    y + r if ay == 0 else y - r,
                    start=[180, 270, 90, 0][int(ax*2 + ay)],
                    extent=90,
                    fill=bg_color,
                    outline=bg_color,
                    tags="corners",
                )
        canvas.bind("<Configure>", lambda e: redraw())
        redraw()

    def _build_preview(self) -> None:
        self.preview_card = ctk.CTkFrame(
            self.content,
            fg_color=self.PANEL,
            corner_radius=self.CORNER_RADIUS,
            border_width=1,
            border_color="#1a3a7a",
        )
        self.preview_card.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.preview_card,
            text="Изображение",
            font=("Arial Bold", 24),
            text_color=self.TEXT,
        )
        title.pack(anchor="w", padx=24, pady=20)

        outer = ctk.CTkFrame(
            self.preview_card,
            fg_color=self.INPUT,
            corner_radius=self.CORNER_RADIUS,
        )
        outer.pack(expand=True, fill="both", padx=20, pady=(0, 24))

        self.preview_canvas = tk.Canvas(
            outer,
            bg=self.INPUT,
            highlightthickness=0,
            borderwidth=0,
        )

        v_scroll = RoundScrollbar(outer, orient="vertical",
            command=self.preview_canvas.yview, width=8)
        h_scroll = RoundScrollbar(outer, orient="horizontal",
            command=self.preview_canvas.xview, height=8)

        self.preview_canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )

        h_scroll.pack(side="bottom", fill="x", padx=16, pady=(0, 8))
        v_scroll.pack(side="right", fill="y", pady=16, padx=(0, 8))
        self.preview_canvas.pack(expand=True, fill="both", padx=(16, 0), pady=16)
        self._draw_canvas_corners(self.preview_canvas, self.INPUT)
        self.preview_canvas.bind("<MouseWheel>", lambda e: self.preview_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.preview_canvas.bind("<Shift-MouseWheel>", lambda e: self.preview_canvas.xview_scroll(-1 * (e.delta // 120), "units"))

    def _build_result(self) -> None:
        self.result_card = ctk.CTkFrame(
            self.content,
            fg_color=self.PANEL,
            corner_radius=self.CORNER_RADIUS,
            border_width=1,
            border_color="#1a3a7a",
        )
        self.result_card.grid(row=2, column=0, sticky="nsew")
        self.result_card.grid_rowconfigure(1, weight=1)
        self.result_card.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.result_card,
            text="Распознанный текст",
            font=("Arial Bold", 24),
            text_color=self.TEXT,
        )
        title.pack(anchor="w", padx=24, pady=20)

        self.result = ctk.CTkTextbox(
            self.result_card,
            fg_color=self.INPUT,
            font=("Arial", 16),
            corner_radius=28,
            border_width=1,
            border_color="#1a3a7a",
        )
        self.result.pack(expand=True, fill="both", padx=20)

        self.copy_btn = ctk.CTkButton(
            self.result_card,
            text="Копировать текст",
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            border_width=1,
            border_color="#2a52c0",
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
                font=("Arial Bold", 16),
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
            font=("Arial Bold", 16),
            fg_color=self.BTN,
            hover_color=self.BTN_HOVER,
            border_width=1,
            border_color="#2a52c0",
            command=cmd,
            corner_radius=28,
        )

    def _on_model_status_click(self, event=None):
        if not self._model_path.exists():
            self.choose_model_file()

    def choose_model_file(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch модели", "*.pt *.pth"), ("Все файлы", "*.*")]
        )
        if file_path:
            self._model_path = Path(file_path)
            self._init_ocr()

    def recognize(self) -> None:
        if not self._ocr:
            self.update_status("Модель не загружена - выберите нажав на линию прогресса", self.RED)
            return
        if not self._image_path:
            self.update_status("Сначала загрузите изображение", self.RED)
            return

        self._processing_animation_active = True
        self._processing_dots = 0
        self._glow_phase = 0.0
        self.update_status("Разглядываю изображение...", self.TEXT)
        self._animate_processing()
        self._animate_progress_glow()
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

    def _lerp_color(self, c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _animate_progress_glow(self) -> None:
        if not self._processing_animation_active:
            try:
                self.progress_container.configure(fg_color=self.BTN, border_color="#2a52c0")
            except Exception:
                pass
            return
        t = (math.sin(self._glow_phase) + 1) / 2
        self._glow_phase = (self._glow_phase + 0.12) % (2 * math.pi)
        color = self._lerp_color(self.BTN, "#3060e0", t)
        border = self._lerp_color("#2a52c0", "#4a7aff", t)
        try:
            self.progress_container.configure(fg_color=color, border_color=border)
        except Exception:
            pass
        self.after(40, self._animate_progress_glow)

    def _on_history_resize(self, event):
        self._history_canvas.configure(scrollregion=self._history_canvas.bbox("all"))

    def _on_history_canvas_resize(self, event):
        self._history_canvas.itemconfig(self._history_canvas_window, width=event.width)

    def _add_history_item(self, text: str) -> None:
        preview = text.strip()[:60].replace("\n", " ")
        if len(text.strip()) > 60:
            preview += "..."

        item_frame = tk.Frame(self._history_frame, bg=self.INPUT)
        item_frame.pack(fill="x", padx=4, pady=4)

        item_canvas = tk.Canvas(
            item_frame,
            bg=self.INPUT,
            highlightthickness=0,
            borderwidth=0,
            height=64,
        )
        item_canvas.pack(fill="x")

        def draw_item(event, canvas=item_canvas, t=preview):
            w = canvas.winfo_width()
            canvas.delete("all")
            r = 16
            canvas.create_polygon(
                r, 0, w-r, 0, w, 0, w, r,
                w, 64-r, w, 64, w-r, 64,
                r, 64, 0, 64, 0, 64-r,
                0, r, 0, 0,
                smooth=True, fill="#1e3a6e",
            )
            canvas.create_text(12, 32, text=t, fill=self.TEXT,
                font=("Arial", 11), anchor="w", width=w-24)

        def on_click(event, full=text):
            self._show_result(full)

        def on_enter(event, canvas=item_canvas, t=preview):
            w = canvas.winfo_width()
            canvas.delete("all")
            r = 16
            canvas.create_polygon(
                r, 0, w-r, 0, w, 0, w, r,
                w, 64-r, w, 64, w-r, 64,
                r, 64, 0, 64, 0, 64-r,
                0, r, 0, 0,
                smooth=True, fill="#2a50cf",
            )
            canvas.create_text(12, 32, text=t, fill="#ffffff",
                font=("Arial", 11), anchor="w", width=w-24)

        def on_leave(event, canvas=item_canvas, t=preview):
            w = canvas.winfo_width()
            canvas.delete("all")
            r = 16
            canvas.create_polygon(
                r, 0, w-r, 0, w, 0, w, r,
                w, 64-r, w, 64, w-r, 64,
                r, 64, 0, 64, 0, 64-r,
                0, r, 0, 0,
                smooth=True, fill="#1e3a6e",
            )
            canvas.create_text(12, 32, text=t, fill=self.TEXT,
                font=("Arial", 11), anchor="w", width=w-24)

        item_canvas.bind("<Configure>", draw_item)
        item_canvas.bind("<Button-1>", on_click)
        item_canvas.bind("<Enter>", on_enter)
        item_canvas.bind("<Leave>", on_leave)

        self._history_items.append(item_frame)
        self._history_canvas.yview_moveto(1.0)

    def _show_result(self, text: str) -> None:
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.update_status("Текст успешно распознан", self.GREEN)
        self._add_history_item(text)

    def copy_text(self) -> None:
        text = self.result.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_status("Текст скопирован в буфер", self.GREEN)

    def clear_all(self) -> None:
        self.preview_canvas.delete("all")
        self._canvas_image_ref = None
        self._image_path = None
        self.result.delete("1.0", "end")
        self.update_status("Жду новое изображение", self.GREEN)

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        self._image_path = Path(file_path)
        image = Image.open(file_path).convert("RGB")

        radius = min(32, min(image.size) // 8)
        rounded = self._make_rounded_image(image, radius=radius)

        self._canvas_image_ref = ImageTk.PhotoImage(rounded)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self._canvas_image_ref, anchor="nw")
        self.preview_canvas.configure(scrollregion=(0, 0, rounded.width, rounded.height))

        self.update_status("Изображение загружено", self.TEXT)

    def _init_ocr(self) -> None:
        if not self._model_path.exists():
            self.update_status(
                f"Модель не найдена: {self._model_path.name} — нажмите сюда и выберите файл модели",
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
