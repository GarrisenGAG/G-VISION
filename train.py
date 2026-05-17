#!/usr/bin/env python3
import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from numba import jit
except ImportError:
    def jit(**kwargs):
        def decorator(func):
            return func
        return decorator

from PIL import Image, ImageFilter


class SystemLogger:
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")


@dataclass
class SessionParameters:
    base_dir: Path = Path("DataSet/Train")
    cache_path: Path = Path("DataSet/dataset_cache_clean.pt")
    img_height: int = 64
    max_width: int = 1024
    batch_size: int = 32
    gradient_accumulation: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    use_amp: bool = False
    use_compile: bool = False
    val_frequency: int = 1
    resume: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    epochs: int = 100
    patience: int = 20
    warmup_epochs: int = 5
    max_samples: int = 3_003_000
    aug_prob: float = 0.60
    aug_rotate: float = 4.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Path = Path("runs/ocr_final")
    model_name: str = "best_model.pt"
    log_file: str = "training_log.txt"
    seed: int = 42


class SymbolEncoder:
    def __init__(self):
        self.vocabulary = ["<blank>", " "] + \
            list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + \
            list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ") + \
            list("0123456789") + \
            list(".,!?;:-—()[]«»\"'/@#№$%&*+=<>~^_{}|\\")
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.size = len(self.vocabulary)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices: List[int], merge_repeats: bool = True, skip_blank: bool = True) -> str:
        output = []
        prev = -1
        for idx in indices:
            if merge_repeats and idx == prev:
                continue
            if skip_blank and idx == 0:
                prev = idx
                continue
            output.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(output)


class AlignmentEvaluator:
    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return AlignmentEvaluator.levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    @staticmethod
    @jit(nopython=True)
    def _levenshtein_fast(s1_codes, s2_codes):
        len1, len2 = len(s1_codes), len(s2_codes)
        if len1 < len2:
            s1_codes, s2_codes = s2_codes, s1_codes
            len1, len2 = len2, len1
        if len2 == 0:
            return len1
        prev = np.arange(len2 + 1, dtype=np.int32)
        curr = np.zeros(len2 + 1, dtype=np.int32)
        for i in range(len1):
            curr[0] = i + 1
            for j in range(len2):
                cost = 0 if s1_codes[i] == s2_codes[j] else 1
                curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
            prev, curr = curr, prev
        return prev[len2]

    @staticmethod
    def compute_distance(s1: str, s2: str) -> float:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return float(len(s1))
        if len(s1) < 5 and len(s2) < 5:
            return float(AlignmentEvaluator.levenshtein(s1, s2))
        try:
            s1_codes = np.array([ord(c) for c in s1], dtype=np.int32)
            s2_codes = np.array([ord(c) for c in s2], dtype=np.int32)
            return float(AlignmentEvaluator._levenshtein_fast(s1_codes, s2_codes))
        except Exception:
            return float(AlignmentEvaluator.levenshtein(s1, s2))

    @staticmethod
    def calculate_error_rate(prediction: str, target: str) -> float:
        if not target:
            return 0.0 if not prediction else 1.0
        return AlignmentEvaluator.compute_distance(prediction, target) / len(target)


class VisualAugmentor:
    def __init__(self, probability: float = 0.60, rotation_limit: float = 4.0):
        self.probability = probability
        self.rotation_limit = rotation_limit
        self.blur_filter = ImageFilter.GaussianBlur

    @staticmethod
    @jit(nopython=True)
    def _adjust_brightness(arr, factor):
        return np.clip(arr * factor, 0.0, 1.0)

    @staticmethod
    @jit(nopython=True)
    def _adjust_contrast(arr, factor):
        mean = np.mean(arr)
        adjusted = (arr - mean) * factor + mean
        return np.clip(adjusted, 0.0, 1.0)

    @staticmethod
    @jit(nopython=True)
    def _apply_noise(arr, std):
        noise = np.random.randn(arr.shape[0], arr.shape[1]) * std
        return np.clip(arr + noise, 0.0, 1.0)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        if random.random() < 0.40:
            arr = np.array(image).astype(np.float32) / 255.0
            arr = self._adjust_brightness(arr, random.uniform(0.70, 1.30))
            image = Image.fromarray((arr * 255).astype(np.uint8))
        if random.random() < 0.40:
            arr = np.array(image).astype(np.float32) / 255.0
            arr = self._adjust_contrast(arr, random.uniform(0.70, 1.30))
            image = Image.fromarray((arr * 255).astype(np.uint8))
        if random.random() < 0.25:
            image = image.filter(self.blur_filter(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.35:
            image = image.rotate(random.uniform(-self.rotation_limit, self.rotation_limit), fillcolor=255, expand=False)
        if random.random() < 0.15:
            arr = np.array(image).astype(np.float32) / 255.0
            arr = self._apply_noise(arr, 0.03)
            image = Image.fromarray((arr * 255).astype(np.uint8))
        return image


class ImageSequenceDataset(Dataset):
    def __init__(self, paths: List[Path], texts: List[str], encoder: SymbolEncoder, apply_augmentation: bool = False, params: Optional[SessionParameters] = None):
        self.paths = paths
        self.texts = texts
        self.encoder = encoder
        self.apply_augmentation = apply_augmentation
        self.transformer = VisualAugmentor(params.aug_prob, params.aug_rotate) if apply_augmentation else None
        self.target_height = params.img_height
        self.max_width = params.max_width

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.paths[idx]
        text = self.texts[idx]
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            img = Image.new("L", (128, self.target_height), 255)
        if self.apply_augmentation and self.transformer:
            img = self.transformer(img)
        width, height = img.size
        if height != self.target_height:
            width = max(32, int(round(width * (self.target_height / height))))
            img = img.resize((width, self.target_height), Image.Resampling.LANCZOS)
        if img.width > self.max_width:
            img = img.resize((self.max_width, self.target_height), Image.Resampling.LANCZOS)
        tensor = transforms.ToTensor()(img)
        tensor = 1.0 - tensor
        ids = torch.tensor(self.encoder.encode(text), dtype=torch.long)
        return {"img": tensor, "ids": ids, "text": text, "len": len(ids)}


def assemble_batch(batch: List[Dict[str, Any]], params: SessionParameters) -> Dict[str, Any]:
    max_width = min(max(item["img"].shape[-1] for item in batch), params.max_width)
    batch_size = len(batch)
    imgs = torch.zeros(batch_size, 1, params.img_height, max_width)
    lengths = []
    texts = []
    for i, item in enumerate(batch):
        width = item["img"].shape[-1]
        imgs[i, :, :, :width] = item["img"]
        lengths.append(item["len"])
        texts.append(item["text"])
    return {
        "imgs": imgs,
        "ids": torch.cat([item["ids"] for item in batch]),
        "lens": torch.tensor(lengths, dtype=torch.long),
        "texts": texts
    }


class SpatialLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, padding: int = 1, pool: Optional[Tuple[int, int]] = None):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class SequenceAligner(nn.Module):
    def __init__(self, output_dim: int, hidden_size: int = 320, use_final_conv: bool = True, legacy: bool = False):
        super().__init__()
        layers = [
            SpatialLayer(1, 64, pool=(2, 2)),
            ResidualUnit(64),
            SpatialLayer(64, 128, pool=(2, 2)),
            ResidualUnit(128),
            SpatialLayer(128, 256, pool=(2, 1)),
            ResidualUnit(256),
            SpatialLayer(256, 512, pool=(2, 1)),
        ]
        if legacy:
            layers.append(SpatialLayer(512, 512, kernel=2, padding=0))
        else:
            layers.append(ResidualUnit(512))
            if use_final_conv:
                layers.append(SpatialLayer(512, 512, kernel=2, padding=0))
        self.extractor = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.recurrence = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.recurrence(x)
        x = self.projection(x)
        return x.permute(1, 0, 2).log_softmax(2)


class PipelineController:
    def __init__(self, aligner: nn.Module, device: torch.device, steps_per_epoch: int, params: SessionParameters):
        self.aligner = aligner
        self.device = device
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.scaler = GradScaler(enabled=(self.device.type == "cuda" and params.use_amp))
        self.optimizer = torch.optim.AdamW(aligner.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
        warmup_epochs = min(params.warmup_epochs, max(1, params.epochs - 1))
        pct_start = max(0.1, min(0.95, warmup_epochs / max(1, params.epochs)))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=params.learning_rate, epochs=params.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=pct_start, div_factor=10, final_div_factor=100
        )
        self.best_metric = 1.0
        self.patience_counter = 0
        self.params = params

    def execute_training_pass(self, loader: DataLoader) -> float:
        self.aligner.train()
        total_loss = 0.0
        total_samples = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc="Training", ncols=100, leave=False), start=1):
            x = batch["imgs"].to(self.device, non_blocking=True)
            y = batch["ids"].to(self.device, non_blocking=True)
            lengths = batch["lens"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda" and self.params.use_amp)):
                logits = self.aligner(x)
                input_lengths = torch.full((x.size(0),), logits.size(0), dtype=torch.long, device=self.device)
                loss = self.loss_fn(logits, y, input_lengths, lengths)

            if torch.isnan(loss):
                self.optimizer.zero_grad()
                continue

            loss = loss / self.params.gradient_accumulation
            self.scaler.scale(loss).backward()

            if step % self.params.gradient_accumulation == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.aligner.parameters(), self.params.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.params.gradient_accumulation * x.size(0)
            total_samples += x.size(0)

        return total_loss / max(total_samples, 1)

    def execute_evaluation_pass(self, loader: DataLoader, encoder: SymbolEncoder) -> Tuple[float, float]:
        self.aligner.eval()
        total_loss = 0.0
        metrics = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluation", ncols=100, leave=False):
                x = batch["imgs"].to(self.device, non_blocking=True)
                y = batch["ids"].to(self.device, non_blocking=True)
                lengths = batch["lens"].to(self.device, non_blocking=True)

                logits = self.aligner(x).float()
                input_lengths = torch.full((x.size(0),), logits.size(0), dtype=torch.long, device=self.device)
                loss = self.loss_fn(logits, y, input_lengths, lengths)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(2).cpu().numpy().T
                for seq, target in zip(preds, batch["texts"]):
                    decoded = encoder.decode(seq.tolist())
                    metrics.append(AlignmentEvaluator.calculate_error_rate(decoded, target))

        return total_loss / max(1, len(loader.dataset)), float(np.mean(metrics))

    def persist_state(self, path: Path, metrics: Dict[str, Any], is_best: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"aligner": self.aligner.state_dict(), "metrics": metrics, "best": is_best}, path)

    def persist_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch, "aligner_state": self.aligner.state_dict(),
            "opt_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "best_metric": self.best_metric, "patience": self.patience_counter
        }, path)

    def restore_checkpoint(self, path: Path) -> int:
        state = torch.load(path, map_location=self.device)
        self.aligner.load_state_dict(state["aligner_state"])
        self.optimizer.load_state_dict(state["opt_state"])
        self.scheduler.load_state_dict(state["sched_state"])
        self.best_metric = state.get("best_metric", self.best_metric)
        self.patience_counter = state.get("patience", self.patience_counter)
        return state.get("epoch", 0)


class Orchestrator:
    def __init__(self):
        self.params = SessionParameters()
        self.logger = SystemLogger()
        self.encoder = SymbolEncoder()

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Sequence alignment and processing pipeline")
        parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
        parser.add_argument("--infer-path", type=str, nargs="+", default=None)
        parser.add_argument("--sample-count", type=int, default=10)
        parser.add_argument("--resume", action="store_true", default=True)
        parser.add_argument("--no-resume", action="store_false", dest="resume")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--max-samples", type=int, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--val-frequency", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        return parser.parse_args()

    def apply_arguments(self, args: argparse.Namespace) -> None:
        if args.infer_path:
            args.infer_path = " ".join(args.infer_path)
        self.params.resume = args.resume
        if args.epochs is not None:
            self.params.epochs = args.epochs
        if args.batch_size is not None:
            self.params.batch_size = args.batch_size
        if args.max_samples is not None:
            self.params.max_samples = args.max_samples
        if args.num_workers is not None:
            self.params.num_workers = args.num_workers
        if args.learning_rate is not None:
            self.params.learning_rate = args.learning_rate
        if args.val_frequency is not None:
            self.params.val_frequency = args.val_frequency
        if args.seed is not None:
            self.params.seed = args.seed

    def setup_environment(self) -> torch.device:
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed_all(self.params.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

        device = torch.device(self.params.device)
        if device.type == "cuda":
            try:
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                self.logger(f"CUDA initialization failed: {e}. Falling back to CPU.")
                device = torch.device("cpu")
        return device

    def load_data(self) -> Tuple[List[Path], List[str]]:
        if not self.params.cache_path.exists():
            self.logger("Cache file not found. Please generate dataset cache first.")
            sys.exit(1)
        cached = torch.load(self.params.cache_path, weights_only=False)
        paths, texts = cached["image_paths"], cached["texts"]
        if len(paths) > self.params.max_samples:
            selected = random.sample(range(len(paths)), self.params.max_samples)
            paths = [paths[i] for i in selected]
            texts = [texts[i] for i in selected]
        return paths, texts

    def prepare_loaders(self, paths: List[Path], texts: List[str], device: torch.device) -> Tuple[DataLoader, DataLoader]:
        indices = list(range(len(paths)))
        random.shuffle(indices)
        n_train = int(len(paths) * 0.90)
        n_val = int(len(paths) * 0.05)
        train_paths = [paths[i] for i in indices[:n_train]]
        train_texts = [texts[i] for i in indices[:n_train]]
        val_paths = [paths[i] for i in indices[n_train:n_train + n_val]]
        val_texts = [texts[i] for i in indices[n_train:n_train + n_val]]

        train_ds = ImageSequenceDataset(train_paths, train_texts, self.encoder, True, self.params)
        val_ds = ImageSequenceDataset(val_paths, val_texts, self.encoder, False, self.params)

        loader_args = {
            "collate_fn": lambda b: assemble_batch(b, self.params),
            "pin_memory": device.type == "cuda",
            "persistent_workers": self.params.num_workers > 0,
        }
        if self.params.num_workers > 0:
            loader_args["prefetch_factor"] = self.params.prefetch_factor

        train_loader = DataLoader(train_ds, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, drop_last=True, **loader_args)
        val_loader = DataLoader(val_ds, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, drop_last=False, **loader_args)
        return train_loader, val_loader

    def initialize_aligner(self, device: torch.device, weights: Optional[Dict] = None) -> SequenceAligner:
        legacy = False
        use_final_conv = True
        hidden_size = 320
        if weights:
            if "rnn.weight_ih_l0" in weights and weights["rnn.weight_ih_l0"].shape[0] == 1024:
                legacy = True
            if "classifier.2.weight" in weights and weights["classifier.2.weight"].shape[1] == 512:
                legacy = True
            if "cnn.7.block.0.weight" in weights or "cnn.7.block.1.weight" in weights:
                use_final_conv = True
            elif "cnn.8.block.0.weight" in weights or "cnn.8.block.1.weight" in weights:
                use_final_conv = True
            hidden_size = 256 if legacy else 320

        aligner = SequenceAligner(self.encoder.size, hidden_size, use_final_conv, legacy).to(device)
        if weights:
            compatible = {k: v for k, v in weights.items() if k in aligner.state_dict() and v.shape == aligner.state_dict()[k].shape}
            aligner.load_state_dict(compatible, strict=False)
        return aligner

    def run_inference(self, aligner: SequenceAligner, image_path: str, device: torch.device) -> str:
        aligner.eval()
        img = Image.open(image_path).convert("L")
        width, height = img.size
        if height != self.params.img_height:
            width = max(32, int(round(width * (self.params.img_height / height))))
            img = img.resize((width, self.params.img_height), Image.Resampling.LANCZOS)
        if img.width > self.params.max_width:
            img = img.resize((self.params.max_width, self.params.img_height), Image.Resampling.LANCZOS)
        tensor = transforms.ToTensor()(img)
        tensor = (1.0 - tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = aligner(tensor)
        seq = logits.argmax(2).cpu().numpy().T[0]
        return self.encoder.decode(seq.tolist())

    def execute(self) -> None:
        args = self.parse_arguments()
        self.apply_arguments(args)
        self.logger("=" * 60)
        self.logger("Sequence Alignment Pipeline Started")
        self.logger(f"Mode: {args.mode}")
        self.logger("=" * 60)

        if torch.cuda.is_available():
            self.logger(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger("CPU mode active")

        self.params.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = SystemLogger(self.params.save_dir / self.params.log_file)

        device = self.setup_environment()
        paths, texts = self.load_data()
        self.logger(f"Loaded {len(paths):,} valid samples")
        if len(paths) > self.params.max_samples:
            self.logger(f"Restricted to {len(paths):,} samples for processing speed")

        train_loader, val_loader = self.prepare_loaders(paths, texts, device)
        checkpoint_path = self.params.save_dir / "checkpoint.pt"
        best_model_path = self.params.save_dir / self.params.model_name

        weights = None
        if self.params.resume and best_model_path.exists() and not checkpoint_path.exists():
            state = torch.load(best_model_path, map_location=device, weights_only=False)
            weights = state.get("aligner", state)

        aligner = self.initialize_aligner(device, weights)
        controller = PipelineController(aligner, device, len(train_loader), self.params)

        start_epoch = 1
        if self.params.resume and checkpoint_path.exists():
            loaded_epoch = controller.restore_checkpoint(checkpoint_path)
            start_epoch = min(loaded_epoch + 1, self.params.epochs)
            self.logger(f"Resumed from checkpoint: epoch {loaded_epoch}, best metric={controller.best_metric:.4f}")
        elif self.params.resume and best_model_path.exists():
            self.logger(f"Loaded weights from {best_model_path} for continuation")

        if args.mode == "eval":
            _, val_metric = controller.execute_evaluation_pass(val_loader, self.encoder)
            self.logger(f"Evaluation complete. Metric: {val_metric:.4f}")
            examples = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["imgs"].to(device, non_blocking=True)
                    logits = aligner(x).float()
                    preds = logits.argmax(2).cpu().numpy().T
                    for seq, target in zip(preds, batch["texts"]):
                        if len(examples) >= args.sample_count:
                            break
                        examples.append((target, self.encoder.decode(seq.tolist())))
                    if len(examples) >= args.sample_count:
                        break
            for idx, (target, pred) in enumerate(examples, 1):
                self.logger(f"  {idx}. Ground: {target}")
                self.logger(f"     Pred:   {pred}")
            return

        if args.mode == "infer":
            if not args.infer_path:
                self.logger("Please provide --infer-path for inference mode")
                sys.exit(1)
            pred = self.run_inference(aligner, args.infer_path, device)
            self.logger(f"Inference: {Path(args.infer_path).name} -> {pred}")
            return

        if self.params.use_compile and hasattr(torch, "compile"):
            try:
                import triton
                aligner = torch.compile(aligner, backend="inductor", mode="reduce-overhead")
                controller.aligner = aligner
                self.logger("Compilation enabled")
            except ImportError:
                self.logger("Triton not available. Compilation disabled.")
            except Exception as exc:
                self.logger(f"Compilation failed: {exc}")
                self.params.use_compile = False

        self.logger(f"Parameters: {sum(p.numel() for p in controller.aligner.parameters()):,}")
        self.logger(f"Starting session: epochs={self.params.epochs}, batch={self.params.batch_size * self.params.gradient_accumulation}, lr={self.params.learning_rate}, resume_from={start_epoch}")

        for epoch in range(start_epoch, self.params.epochs + 1):
            start_time = time.time()
            train_loss = controller.execute_training_pass(train_loader)
            elapsed = time.time() - start_time

            if epoch == 1 or epoch % self.params.val_frequency == 0:
                val_loss, val_metric = controller.execute_evaluation_pass(val_loader, self.encoder)
                self.logger(f"Epoch {epoch}/{self.params.epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Metric={val_metric:.4f} | Time={elapsed/60:.1f}m")

                if val_metric < controller.best_metric:
                    controller.best_metric = val_metric
                    controller.patience_counter = 0
                    controller.persist_state(self.params.save_dir / self.params.model_name, {"metric": val_metric, "epoch": epoch}, is_best=True)
                    self.logger(f"New best metric: {val_metric:.4f}")
                else:
                    controller.patience_counter += 1
                    if controller.patience_counter >= self.params.patience:
                        self.logger(f"Early stopping at epoch {epoch}")
                        break
            else:
                self.logger(f"Epoch {epoch}/{self.params.epochs} | Train Loss={train_loss:.4f} | Val=n/a | Metric=n/a | Time={elapsed/60:.1f}m")

            if device.type == "cuda":
                torch.cuda.empty_cache()

        self.logger("=" * 60)
        self.logger(f"Session complete. Best metric: {controller.best_metric:.4f}")
        self.logger("=" * 60)


if __name__ == "__main__":
    Orchestrator().execute()
