import sys

def resource_path(relative_path: str) -> Path:
    try:
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        base_path = Path(__file__).parent
    return base_path / relative_path
import math
import platform
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple
import tkinter as tk
import json
from datetime import datetime

import customtkinter as ctk
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms

warnings.filterwarnings("ignore", category=DeprecationWarning)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


@dataclass
class InferenceConfig:

    img_height: int = 64
    max_width: int = 1024


class SymbolEncoder:

    VOCABULARY: List[str] = (
        ["<blank>", " "]
        + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        + list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
        + list("0123456789")
        + list(".,!?;:-—()[]«»\"'/@#№$%&*+=<>~^_{}|\\")
    )

    def __init__(self) -> None:
        self._char_to_idx: Dict[str, int] = {
            ch: i for i, ch in enumerate(self.VOCABULARY)
        }
        self._idx_to_char: Dict[int, str] = {
            i: ch for ch, i in self._char_to_idx.items()
        }
        self.size: int = len(self.VOCABULARY)

    def encode(self, text: str) -> List[int]:
        return [self._char_to_idx.get(ch, 0) for ch in text]

    def decode(
        self,
        indices: List[int],
        merge_repeats: bool = True,
        skip_blank: bool = True,
    ) -> str:
        output: List[str] = []
        prev = -1
        for idx in indices:
            if merge_repeats and idx == prev:
                continue
            if skip_blank and idx == 0:
                prev = idx
                continue
            output.append(self._idx_to_char.get(idx, ""))
            prev = idx
        return "".join(output)


class SpatialLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        padding: int = 1,
        pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels, kernel,
                padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class SequenceAligner(nn.Module):

    def __init__(
        self,
        output_dim: int,
        hidden_size: int = 320,
        use_final_conv: bool = True,
        legacy: bool = False,
    ) -> None:
        super().__init__()

        cnn_layers: List[nn.Module] = [
            SpatialLayer(1, 64, pool=(2, 2)),
            ResidualUnit(64),
            SpatialLayer(64, 128, pool=(2, 2)),
            ResidualUnit(128),
            SpatialLayer(128, 256, pool=(2, 1)),
            ResidualUnit(256),
            SpatialLayer(256, 512, pool=(2, 1)),
        ]
        if legacy:
            cnn_layers.append(SpatialLayer(512, 512, kernel=2, padding=0))
        else:
            cnn_layers.append(ResidualUnit(512))
            if use_final_conv:
                cnn_layers.append(SpatialLayer(512, 512, kernel=2, padding=0))

        self.extractor = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.recurrence = nn.LSTM(
            512,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.recurrence(x)
        x = self.projection(x)
        return x.permute(1, 0, 2).log_softmax(2)


_LEGACY_KEY_MAP: Dict[str, str] = {
    "cnn.": "extractor.",
    "rnn.": "recurrence.",
    "classifier.": "projection.",
}


class GVisionOCR:

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_compile: bool = False,
    ) -> None:
        self._config = InferenceConfig()
        self._encoder = SymbolEncoder()

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device == "auto"
            else device
        )

        state = torch.load(
            model_path, map_location=self._device, weights_only=False
        )
        weights: Dict = state.get("aligner", state)
        weights = {self._remap_key(k): v for k, v in weights.items()}

        legacy = (
            "recurrence.weight_ih_l0" in weights
            and weights["recurrence.weight_ih_l0"].shape[0] == 1024
        )
        hidden_size = 256 if legacy else 320

        self._model = SequenceAligner(
            self._encoder.size,
            hidden_size,
            use_final_conv=True,
            legacy=legacy,
        ).to(self._device)

        model_state = self._model.state_dict()
        compatible = {
            k: v
            for k, v in weights.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        self._model.load_state_dict(compatible, strict=False)
        self._model.eval()

        if use_compile and hasattr(torch, "compile"):
            try:
                self._model = torch.compile(self._model)
            except Exception:
                pass

    @staticmethod
    def _remap_key(key: str) -> str:
        for old, new in _LEGACY_KEY_MAP.items():
            if key.startswith(old):
                return new + key[len(old):]
        return key

    def recognize(self, image_path: str) -> Dict[str, str]:
        img = Image.open(image_path).convert("L")
        width, height = img.size
        target_h = self._config.img_height

        if height != target_h:
            new_w = max(32, int(round(width * target_h / height)))
            img = img.resize((new_w, target_h), Image.Resampling.LANCZOS)
        if img.width > self._config.max_width:
            img = img.resize(
                (self._config.max_width, target_h), Image.Resampling.LANCZOS
            )

        tensor = transforms.ToTensor()(img)
        tensor = (1.0 - tensor).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)

        seq = logits.argmax(2).cpu().numpy().T[0]
        return {"text": self._encoder.decode(seq.tolist())}


class RoundScrollbar(tk.Canvas):

    def __init__(
        self,
        parent,
        orient: str = "vertical",
        command=None,
        bg: str = "#14306a",
        thumb_color: str = "#1e40af",
        hover_color: str = "#2a50cf",
        **kwargs,
    ) -> None:
        super().__init__(
            parent, bg=bg, highlightthickness=0, borderwidth=0, **kwargs
        )
        self._orient = orient
        self._command = command
        self._thumb_color = thumb_color
        self._hover_color = hover_color
        self._pos: Tuple[float, float] = (0.0, 1.0)
        self._dragging = False
        self._drag_start: Optional[int] = None

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self._draw(hover=True))
        self.bind("<Leave>", lambda e: self._draw(hover=False))
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, lo: str, hi: str) -> None:
        self._pos = (float(lo), float(hi))
        self._draw()

    def _draw(self, hover: bool = False) -> None:
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
            r = max((x1 - x0) // 2, 4)
        else:
            pad = 2
            y0, y1 = pad, h - pad
            x0 = lo * w + pad
            x1 = hi * w - pad
            r = max((y1 - y0) // 2, 4)

        self._draw_rounded_rect(x0, y0, x1, y1, r, fill=color)

    def _draw_rounded_rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        r: float,
        **kwargs,
    ) -> None:
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
            smooth=True,
            **kwargs,
        )

    def _on_press(self, event) -> None:
        self._dragging = True
        self._drag_start = (
            event.y if self._orient == "vertical" else event.x
        )

    def _on_drag(self, event) -> None:
        if not self._dragging or self._drag_start is None:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        size = h if self._orient == "vertical" else w
        pos = event.y if self._orient == "vertical" else event.x
        delta = (pos - self._drag_start) / size
        self._drag_start = pos
        if self._command:
            self._command("moveto", self._pos[0] + delta)

    def _on_release(self, event) -> None:
        self._dragging = False


class App(ctk.CTk):

    WIDTH: int = 1400
    HEIGHT: int = 920
    HISTORY_FILE: str = "gvision_history.json"
    MODEL_CONFIG_FILE: str = "gvision_model.json"
    MAX_HISTORY_ITEMS: int = 100

    BG: str = "#020208"
    SIDEBAR: str = "#0a1a4b"
    PANEL: str = "#0a1a4b"
    INPUT: str = "#14306a"
    BTN: str = "#1e40af"
    BTN_HOVER: str = "#15307f"
    BTN_BORDER: str = "#2a52c0"
    GREEN: str = "#22c55e"
    RED: str = "#ef4444"
    TEXT: str = "#e6eefc"
    SUBTEXT: str = "#94a3b8"
    CORNER_RADIUS: int = 32
    TRANSPARENT: str = "#010101"

    def __init__(self) -> None:
        super().__init__()

        self._window_rounded: bool = False
        self.overrideredirect(True)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(1100, 760)
        self.title("G-VISION")
        self._set_window_icon() 
        
        if platform.system() == "Windows":
            self.configure(fg_color=self.TRANSPARENT)
        else:
            self.configure(fg_color=self.BG)

        self._apply_window_rounding()
        self._set_window_icon()
        self._image_path: Optional[Path] = None
        self._ocr: Optional[GVisionOCR] = None
        self._drag_position: Optional[Tuple[int, int]] = None
        self._processing_animation_active: bool = False
        self._processing_dots: int = 0
        self._glow_phase: float = 0.0
        self._job_id: int = 0
        self._canvas_image_ref = None
        self._history_items: list = []

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._device: str = "auto"
        self._use_compile: bool = False

        self._load_model_path()
        self._build()
        self._init_ocr()
        self.protocol("WM_DELETE_WINDOW", self.close)


    def _apply_window_rounding(self) -> None:
        system = platform.system()
        if system == "Windows":

            try:
                self.wm_attributes("-transparentcolor", self.TRANSPARENT)
                self._window_rounded = True
            except Exception:
                pass
        elif system == "Darwin":

            try:
                self.wm_attributes("-transparent", True)
                self._window_rounded = True
            except Exception:
                pass

    def get_resource_path(relative: str) -> Path:
        if getattr(sys, 'frozen', False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent
        return base / relative
        
    def _set_window_icon(self) -> None:
        icon_path = resource_path("logoC.ico")
        icon_png = resource_path("logoC.png")
        
        system = platform.system()
        
        if system == "Windows":
            if icon_path.exists():
                try:
                    self.iconbitmap(str(icon_path))
                except Exception:
                    pass
        else:
            icon_candidate = icon_png if icon_png.exists() else icon_path
            if icon_candidate.exists():
                try:
                    icon = tk.PhotoImage(file=str(icon_candidate))
                    self.tk.call("wm", "iconphoto", self._w, icon)
                    self._icon_ref = icon
                except Exception:
                    pass

    def _build(self) -> None:
        system = platform.system()
        if system == "Windows":

            self._bg_frame = ctk.CTkFrame(
                self,
                fg_color=self.BG,
                corner_radius=self.CORNER_RADIUS,
                border_width=2,
                border_color="#081327",
            )
            self._bg_frame.place(x=0, y=0, relwidth=1, relheight=1)
            self._bg_frame.lower()
        elif system == "Darwin":

            self._bg_frame = ctk.CTkFrame(
                self,
                fg_color=self.BG,
                corner_radius=self.CORNER_RADIUS,
                border_width=2,
                border_color="#081327",
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
        self.sidebar.grid(
            row=1, column=0, sticky="ns", padx=(16, 8), pady=(0, 16)
        )

        self.content = ctk.CTkFrame(
            self,
            fg_color=self.BG,
            corner_radius=self.CORNER_RADIUS,
            bg_color=self.BG,
        )
        self.content.grid(
            row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16)
        )
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_preview()
        self._build_result()
        self._load_history()

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
            row=0, column=0, columnspan=2,
            sticky="ew", padx=16, pady=(12, 12),
        )
        header.grid_columnconfigure(0, weight=0)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, weight=0)

        left_frame = ctk.CTkFrame(header, fg_color=self.SIDEBAR, corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="w", padx=(18, 8), pady=12)
        left_frame.grid_columnconfigure(0, weight=0)
        left_frame.grid_columnconfigure(1, weight=0)

        self._build_logo(left_frame)

        title = ctk.CTkLabel(
            left_frame,
            text="G-VISION",
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
            border_color=self.BTN_BORDER,
        )
        self.progress_container.grid(
            row=0, column=1, sticky="ew", padx=8, pady=12
        )
        self.progress_container.grid_propagate(False)
        self.progress_container.configure(height=42)

        self.header_status_label = ctk.CTkLabel(
            self.progress_container,
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
            border_color=self.BTN_BORDER,
            command=self.close,
        )
        self.close_button.grid(row=0, column=2, sticky="e", padx=16, pady=12)

    def _build_logo(self, parent: ctk.CTkFrame) -> None:
        logo_path = Path(__file__).parent / "logo.png"
        if logo_path.exists():
            try:
                scale = 8
                circle_size = 56 * scale
                logo_img = Image.open(logo_path).convert("RGBA")
                logo_img = logo_img.resize(
                    (40 * scale, 40 * scale), Image.LANCZOS
                )
                circle = Image.new("RGBA", (circle_size, circle_size), (0, 0, 0, 0))
                mask = Image.new("L", (circle_size, circle_size), 0)
                ImageDraw.Draw(mask).ellipse(
                    (0, 0, circle_size, circle_size), fill=255
                )
                white = Image.new(
                    "RGBA", (circle_size, circle_size), (255, 255, 255, 255)
                )
                circle.paste(white, mask=mask)
                offset = (
                    (circle_size - logo_img.width) // 2,
                    (circle_size - logo_img.height) // 2,
                )
                circle.paste(logo_img, offset, logo_img)
                circle = circle.resize((56, 56), Image.LANCZOS)
                self._logo_image = ctk.CTkImage(circle, size=(56, 56))
                logo_label = ctk.CTkLabel(
                    parent,
                    image=self._logo_image,
                    text="",
                    fg_color="transparent",
                )
                logo_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
                logo_label.bind("<ButtonPress-1>", self.start_move)
                return
            except Exception:
                pass


        logo_bg = ctk.CTkFrame(
            parent, width=48, height=48,
            fg_color=self.BTN, corner_radius=28,
        )
        logo_bg.grid(row=0, column=0, sticky="w", padx=(0, 8))
        logo_bg.grid_propagate(False)
        logo_label = ctk.CTkLabel(
            logo_bg, text="G",
            font=("Arial Bold", 18),
            text_color="#ffffff",
        )
        logo_label.place(relx=0.5, rely=0.5, anchor="center")
        logo_bg.bind("<ButtonPress-1>", self.start_move)
        logo_label.bind("<ButtonPress-1>", self.start_move)

    def _build_sidebar(self) -> None:
        self.btn_load = self._make_button("Загрузить изображение", self.load_image)
        self.btn_load.pack(fill="x", padx=20, pady=(20, 8))

        self.btn_run = self._make_button("Распознать текст", self.recognize)
        self.btn_run.pack(fill="x", padx=20, pady=8)

        self.btn_clear = self._make_button("Очистить", self.clear_all, red=True)
        self.btn_clear.pack(fill="x", padx=20, pady=(8, 20))

        ctk.CTkLabel(
            self.sidebar,
            text="История",
            font=("Arial Bold", 16),
            text_color=self.SUBTEXT,
        ).pack(anchor="w", padx=24, pady=(0, 8))

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
        self._history_canvas.pack(
            side="left", fill="both", expand=True, padx=(8, 0), pady=8
        )

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
        self._history_canvas.bind(
            "<MouseWheel>",
            lambda e: self._history_canvas.yview_scroll(
                -1 * (e.delta // 120), "units"
            ),
        )

    def _get_history_path(self) -> Path:
        return Path(__file__).parent / self.HISTORY_FILE

    def _get_model_config_path(self) -> Path:
        return resource_path(self.MODEL_CONFIG_FILE)

    def _load_model_path(self) -> None:
        config_path = self._get_model_config_path()
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                    model_path = data.get("model_path")
                    if model_path:
                        self._model_path = Path(model_path)
            except Exception:
                self._model_path = resource_path("G-Vision 13m.pt")
        else:
            self._model_path = resource_path("G-Vision 13m.pt")

    def _save_model_path(self) -> None:
        config_path = self._get_model_config_path()
        try:
            with open(config_path, "w") as f:
                json.dump({"model_path": str(self._model_path)}, f)
        except Exception:
            pass

    def _load_history(self) -> None:
        history_path = self._get_history_path()
        if not history_path.exists():
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in reversed(items[-20:]): 
                self._add_history_item(item["text"], save_to_file=False)
        except Exception as e:
            self.update_status(f"Ошибка загрузки истории: {e}", self.RED)

    def _save_history_item(self, text: str) -> None:
        history_path = self._get_history_path()
        record = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "preview": text.strip()[:60].replace("\n", " ")
        }
        
        items = []
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception:
                items = []

        items.append(record)
        items = items[-self.MAX_HISTORY_ITEMS:]

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _clear_history_file(self) -> None:
        history_path = self._get_history_path()
        if history_path.exists():
            try:
                history_path.unlink()
                for item in self._history_items:
                    item.destroy()
                self._history_items.clear()
            except Exception as e:
                self.update_status(f"Ошибка очистки истории: {e}", self.RED)

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

        ctk.CTkLabel(
            self.preview_card,
            text="Изображение",
            font=("Arial Bold", 24),
            text_color=self.TEXT,
        ).pack(anchor="w", padx=24, pady=20)

        outer = ctk.CTkFrame(
            self.preview_card,
            fg_color=self.INPUT,
            corner_radius=self.CORNER_RADIUS,
        )
        outer.pack(expand=True, fill="both", padx=20, pady=(0, 24))

        self.preview_canvas = tk.Canvas(
            outer, bg=self.INPUT, highlightthickness=0, borderwidth=0
        )
        v_scroll = RoundScrollbar(
            outer, orient="vertical",
            command=self.preview_canvas.yview, width=8,
        )
        h_scroll = RoundScrollbar(
            outer, orient="horizontal",
            command=self.preview_canvas.xview, height=8,
        )
        self.preview_canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )
        h_scroll.pack(side="bottom", fill="x", padx=16, pady=(0, 8))
        v_scroll.pack(side="right", fill="y", pady=16, padx=(0, 8))
        self.preview_canvas.pack(
            expand=True, fill="both", padx=(16, 0), pady=16
        )
        self._draw_canvas_corners(self.preview_canvas, self.INPUT)
        self.preview_canvas.bind(
            "<MouseWheel>",
            lambda e: self.preview_canvas.yview_scroll(
                -1 * (e.delta // 120), "units"
            ),
        )
        self.preview_canvas.bind(
            "<Shift-MouseWheel>",
            lambda e: self.preview_canvas.xview_scroll(
                -1 * (e.delta // 120), "units"
            ),
        )

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

        ctk.CTkLabel(
            self.result_card,
            text="Распознанный текст",
            font=("Arial Bold", 24),
            text_color=self.TEXT,
        ).pack(anchor="w", padx=24, pady=20)

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
            border_color=self.BTN_BORDER,
            command=self.copy_text,
            width=220,
            height=48,
            corner_radius=28,
        )
        self.copy_btn.pack(anchor="e", padx=20, pady=18)


    def _make_button(
        self, text: str, cmd, red: bool = False
    ) -> ctk.CTkButton:
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
            border_color=self.BTN_BORDER,
            command=cmd,
            corner_radius=28,
        )

    @staticmethod
    def _draw_canvas_corners(
        canvas: tk.Canvas, bg_color: str, radius: int = 24
    ) -> None:

        def redraw(event=None) -> None:
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
                    x - r * (1 - 2 * ax),
                    y - r * (1 - 2 * ay),
                    x + r * (2 * ax - 0) if ax == 0 else x - r,
                    y + r if ay == 0 else y - r,
                    start=[180, 270, 90, 0][int(ax * 2 + ay)],
                    extent=90,
                    fill=bg_color,
                    outline=bg_color,
                    tags="corners",
                )

        canvas.bind("<Configure>", lambda e: redraw())
        redraw()


    def _init_ocr(self) -> None:
        if not self._model_path.exists():
            self.update_status(
                f"Модель не найдена: {self._model_path.name}"
                " - нажмите сюда и выберите файл модели",
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
            self.update_status(
                f"Модель загружена: {self._model_path.name}", self.TEXT
            )
        except Exception as error:
            self._ocr = None
            self.update_status(f"Ошибка загрузки OCR: {error}", self.RED)

    def recognize(self) -> None:
        if not self._ocr:
            self.update_status(
                "Модель не загружена — нажмите на линию прогресса", self.RED
            )
            return
        if not self._image_path:
            self.update_status("Сначала загрузите изображение", self.RED)
            return

        self._job_id += 1
        current_job = self._job_id
        self._processing_animation_active = True
        self._glow_phase = 0.0
        self.update_status("Распознавание...", self.TEXT)
        self._animate_progress_glow()
        # форсируем отрисовку статуса до блокировки главного треда
        self.update_idletasks()
        self._run_sync(current_job)

    def _run_sync(self, job_id: int) -> None:
        if job_id != self._job_id:
            self._finish_processing()
            return
        try:
            result = self._ocr.recognize(str(self._image_path))
            text = result.get("text", "")
            self._show_result(text)
        except Exception as error:
            import traceback
            full = traceback.format_exc()
            self.update_status(str(error)[:120], self.RED)
            print(full, flush=True)
        finally:
            self._finish_processing()


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

    @staticmethod
    def _lerp_color(c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return "#{:02x}{:02x}{:02x}".format(
            int(r1 + (r2 - r1) * t),
            int(g1 + (g2 - g1) * t),
            int(b1 + (b2 - b1) * t),
        )

    def _animate_progress_glow(self) -> None:
        if not self._processing_animation_active:
            try:
                self.progress_container.configure(
                    fg_color=self.BTN, border_color=self.BTN_BORDER
                )
            except Exception:
                pass
            return
        t = (math.sin(self._glow_phase) + 1) / 2
        self._glow_phase = (self._glow_phase + 0.12) % (2 * math.pi)
        try:
            self.progress_container.configure(
                fg_color=self._lerp_color(self.BTN, "#3060e0", t),
                border_color=self._lerp_color(self.BTN_BORDER, "#4a7aff", t),
            )
        except Exception:
            pass
        self.after(40, self._animate_progress_glow)


    def _on_history_resize(self, event) -> None:
        self._history_canvas.configure(
            scrollregion=self._history_canvas.bbox("all")
        )

    def _on_history_canvas_resize(self, event) -> None:
        self._history_canvas.itemconfig(
            self._history_canvas_window, width=event.width
        )

    def _add_history_item(self, text: str, save_to_file: bool = True) -> None:
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

        state = {"fill": "#1e3a6e"}

        def draw(canvas: tk.Canvas, label: str, fill: str) -> None:
            state["fill"] = fill
            w = canvas.winfo_width()
            if w < 2:
                return
            canvas.delete("all")
            r = 16
            canvas.create_polygon(
                r, 0, w - r, 0, w, 0, w, r,
                w, 64 - r, w, 64, w - r, 64,
                r, 64, 0, 64, 0, 64 - r,
                0, r, 0, 0,
                smooth=True, fill=fill,
            )
            canvas.create_text(
                12, 32, text=label, fill=self.TEXT,
                font=("Arial", 11), anchor="w", width=w - 24,
            )

        item_canvas.bind(
            "<Configure>",
            lambda e, c=item_canvas, t=preview: draw(c, t, state["fill"]),
        )

        item_canvas.bind(
            "<Button-1>",
            lambda e, full=text: self._display_text(full),
        )
        item_canvas.bind(
            "<Enter>",
            lambda e, c=item_canvas, t=preview: draw(c, t, "#2a50cf"),
        )
        item_canvas.bind(
            "<Leave>",
            lambda e, c=item_canvas, t=preview: draw(c, t, "#1e3a6e"),
        )

        self._history_items.append(item_frame)

        self._history_frame.update_idletasks()
        self._history_canvas.configure(
            scrollregion=self._history_canvas.bbox("all")
        )
        self._history_canvas.yview_moveto(1.0)

        if save_to_file:
            self._save_history_item(text)


    def _display_text(self, text: str) -> None:
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.update_status("Текст успешно распознан", self.GREEN)

    def _show_result(self, text: str) -> None:
        self._display_text(text)
        self._add_history_item(text)

    def copy_text(self) -> None:
        text = self.result.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_status("Текст скопирован в буфер", self.GREEN)

    def clear_all(self) -> None:
        self._job_id += 1
        self._processing_animation_active = False
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
        self.preview_canvas.create_image(
            0, 0, image=self._canvas_image_ref, anchor="nw"
        )
        self.preview_canvas.configure(
            scrollregion=(0, 0, rounded.width, rounded.height)
        )
        self.update_status("Изображение загружено", self.TEXT)

    def choose_model_file(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("PyTorch модели", "*.pt *.pth"),
                ("Все файлы", "*.*"),
            ]
        )
        if file_path:
            self._model_path = Path(file_path)
            self._save_model_path()
            self._init_ocr()

    def _make_rounded_image(
        self, image: Image.Image, radius: int
    ) -> Image.Image:
        image = image.convert("RGBA")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, image.width, image.height), radius=radius, fill=255
        )
        image.putalpha(mask)
        rounded = Image.new("RGBA", image.size, self.INPUT)
        rounded.paste(image, (0, 0), image)
        return rounded


    def update_status(
        self, text: str, text_color: Optional[str] = None
    ) -> None:
        try:
            self.header_status_label.configure(
                text=text,
                text_color=text_color or self.SUBTEXT,
            )
        except Exception:
            pass

    def _on_model_status_click(self, event=None) -> None:
        if not self._model_path.exists():
            self.choose_model_file()


    def start_move(self, event) -> None:
        self._drag_position = (event.x_root, event.y_root)

    def move_window(self, event) -> None:
        if self._drag_position is None:
            return
        delta_x = event.x_root - self._drag_position[0]
        delta_y = event.y_root - self._drag_position[1]
        self.geometry(
            f"+{self.winfo_x() + delta_x}+{self.winfo_y() + delta_y}"
        )
        self._drag_position = (event.x_root, event.y_root)

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
