#!/usr/bin/env python3
"""
G-VISION OCR — Handwritten Optimizer

Rewritten trainer and model architecture for improved handwritten text recognition,
optimized for training speed and recognition quality.
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('high')


@dataclass
class Cfg:
    BASE_DIR: Path = Path("DataSet/Train")
    CACHE_PATH: Path = Path("DataSet/dataset_cache_clean.pt")

    IMG_HEIGHT: int = 64
    MAX_WIDTH: int = 1024

    BATCH_SIZE: int = 16
    GRADIENT_ACCUMULATION: int = 2
    NUM_WORKERS: int = 0
    PREFETCH_FACTOR: int = 2
    USE_AMP: bool = False
    USE_COMPILE: bool = False
    VAL_FREQUENCY: int = 1
    RESUME: bool = True

    LR: float = 6e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    EPOCHS: int = 60
    PATIENCE: int = 10
    WARMUP_EPOCHS: int = 5

    MAX_SAMPLES: int = 150_000

    AUG_PROB: float = 0.75
    AUG_ROTATE: float = 3.0

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: Path = Path("runs/ocr_final")
    MODEL_NAME: str = "best_model.pt"
    LOG_FILE: str = "training_log.txt"
    SEED: int = 42

VOCAB_LIST = ["<blank>", " "] + \
    list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + \
    list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ") + \
    list("0123456789") + \
    list(".,!?;:-—()[]«»\"'/@#№$%&*+=<>~^_{}|\\")

CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB_LIST)}
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}
VOCAB_SIZE = len(VOCAB_LIST)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str, lf: str | None = None) -> None:
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    if lf:
        with open(lf, 'a', encoding='utf-8') as f:
            f.write(f"[{t}] {msg}\n")


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def calc_cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


def ctc_greedy_decode(sequence: np.ndarray) -> str:
    output = []
    prev = None
    for idx in sequence:
        idx = int(idx)
        if idx == prev or idx == 0:
            prev = idx
            continue
        output.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return ''.join(output)


class HandwrittenAugment:
    def __init__(self) -> None:
        self.blur = ImageFilter.GaussianBlur

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > C.AUG_PROB:
            return img

        if random.random() < 0.35:
            img = img.filter(self.blur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.45:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.30))
        if random.random() < 0.45:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.70, 1.30))
        if random.random() < 0.25:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.35:
            img = img.rotate(random.uniform(-C.AUG_ROTATE, C.AUG_ROTATE), fillcolor=255)
        if random.random() < 0.18:
            img = self._add_noise(img)
        if random.random() < 0.20:
            img = ImageOps.autocontrast(img)
        return img

    @staticmethod
    def _add_noise(img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        arr += np.random.randn(*arr.shape) * 0.04
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


class OCRDataset(Dataset):
    def __init__(self, paths: list[Path | str], texts: list[str], augment: bool = False) -> None:
        self.paths = paths
        self.texts = texts
        self.augment = augment
        self.augmentor = HandwrittenAugment()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.paths[idx]
        text = self.texts[idx]

        try:
            img = Image.open(img_path).convert('L')
        except Exception:
            img = Image.new('L', (128, C.IMG_HEIGHT), 255)

        if self.augment:
            img = self.augmentor(img)

        width, height = img.size
        if height != C.IMG_HEIGHT:
            width = max(32, int(round(width * (C.IMG_HEIGHT / height))))
            img = img.resize((width, C.IMG_HEIGHT), Image.Resampling.LANCZOS)
        if img.width > C.MAX_WIDTH:
            img = img.resize((C.MAX_WIDTH, C.IMG_HEIGHT), Image.Resampling.LANCZOS)

        tensor = transforms.ToTensor()(img)
        tensor = 1.0 - tensor

        ids = [CHAR2IDX.get(ch, 0) for ch in text]
        return {
            'img': tensor,
            'ids': torch.tensor(ids, dtype=torch.long),
            'txt': text,
            'len': len(ids)
        }


def collate(batch: list[dict]) -> dict:
    max_width = min(max(item['img'].shape[-1] for item in batch), C.MAX_WIDTH)
    batch_size = len(batch)

    imgs = torch.zeros(batch_size, 1, C.IMG_HEIGHT, max_width)
    ids = []
    lengths = []
    texts = []

    for i, item in enumerate(batch):
        width = item['img'].shape[-1]
        imgs[i, :, :, :width] = item['img']
        ids.append(item['ids'])
        lengths.append(item['len'])
        texts.append(item['txt'])

    return {
        'imgs': imgs,
        'ids': torch.cat(ids),
        'lens': torch.tensor(lengths, dtype=torch.long),
        'txts': texts
    }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, padding: int = 1, pool: tuple[int, int] | None = None) -> None:
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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
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


class CRNN(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int = 320, use_final_conv: bool = True, legacy: bool = False) -> None:
        super().__init__()
        layers = [
            ConvBlock(1, 64, pool=(2, 2)),
            ResidualBlock(64),
            ConvBlock(64, 128, pool=(2, 2)),
            ResidualBlock(128),
            ConvBlock(128, 256, pool=(2, 1)),
            ResidualBlock(256),
            ConvBlock(256, 512, pool=(2, 1)),
        ]
        if legacy:
            layers.append(ConvBlock(512, 512, kernel=2, padding=0))
        else:
            layers.append(ResidualBlock(512))
            if use_final_conv:
                layers.append(ConvBlock(512, 512, kernel=2, padding=0))
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x.permute(1, 0, 2).log_softmax(2)


class Trainer:
    def __init__(self, model: CRNN, device: torch.device, steps_per_epoch: int) -> None:
        self.model = model
        self.device = device
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda' and C.USE_AMP))
        self.opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        pct_start = min(1.0, C.WARMUP_EPOCHS / max(1, C.EPOCHS))
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=C.LR,
            epochs=C.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=10,
            final_div_factor=100
        )
        self.best_cer = 1.0
        self.patience = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        self.opt.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc='Train', ncols=100, leave=False), start=1):
            x = batch['imgs'].to(self.device, non_blocking=True)
            y = batch['ids'].to(self.device, non_blocking=True)
            lengths = batch['lens'].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda' and C.USE_AMP)):
                logits = self.model(x)
                input_lengths = torch.full((x.size(0),), logits.size(0), dtype=torch.long, device=self.device)
                loss = self.ctc(logits, y, input_lengths, lengths)

            if torch.isnan(loss):
                self.opt.zero_grad()
                continue

            loss = loss / C.GRADIENT_ACCUMULATION
            self.scaler.scale(loss).backward()

            if step % C.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad()

            total_loss += loss.item() * C.GRADIENT_ACCUMULATION * x.size(0)
            total_samples += x.size(0)

        return total_loss / max(total_samples, 1)

    def val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        cers = []

        with torch.no_grad():
            for batch in tqdm(loader, desc='Val', ncols=100, leave=False):
                x = batch['imgs'].to(self.device, non_blocking=True)
                y = batch['ids'].to(self.device, non_blocking=True)
                lengths = batch['lens'].to(self.device, non_blocking=True)

                logits = self.model(x).float()
                input_lengths = torch.full((x.size(0),), logits.size(0), dtype=torch.long, device=self.device)
                loss = self.ctc(logits, y, input_lengths, lengths)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(2).cpu().numpy().T
                for seq, target in zip(preds, batch['txts']):
                    pred_text = ctc_greedy_decode(seq)
                    cers.append(calc_cer(pred_text, target))

        return total_loss / max(1, len(loader.dataset)), float(np.mean(cers))

    def save(self, path: Path, metrics: dict, is_best: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'metrics': metrics, 'best': is_best}, path)
        log(f'Model saved to {path}')

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'opt_state': self.opt.state_dict(),
            'sched_state': self.sched.state_dict(),
            'best_cer': self.best_cer,
            'patience': self.patience
        }
        torch.save(checkpoint, path)
        log(f'Checkpoint saved: {path}')

    def load_checkpoint(self, path: Path, device: torch.device) -> int:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt.load_state_dict(checkpoint['opt_state'])
        self.sched.load_state_dict(checkpoint['sched_state'])
        self.best_cer = checkpoint.get('best_cer', self.best_cer)
        self.patience = checkpoint.get('patience', self.patience)
        return checkpoint.get('epoch', 0)


def load_model_weights(path: Path, model: CRNN, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state.get('model', state)

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for key, value in weights.items():
        if key not in model_state:
            skipped.append((key, 'missing'))
            continue
        if value.shape != model_state[key].shape:
            skipped.append((key, f'{value.shape}->{model_state[key].shape}'))
            continue
        loadable[key] = value

    model.load_state_dict(loadable, strict=False)

    if skipped:
        log(f'Warning: {len(skipped)} parameters skipped during loading {path.name}:')
        for key, reason in skipped[:10]:
            log(f'   - {key}: {reason}')
        if len(skipped) > 10:
            log(f'   ... and {len(skipped) - 10} more parameters skipped')
    else:
        log(f'Loaded {len(loadable)} parameters from {path.name}')

    return state


def is_legacy_checkpoint(weights: dict) -> bool:
    if 'rnn.weight_ih_l0' in weights and weights['rnn.weight_ih_l0'].shape[0] == 1024:
        return True
    if 'classifier.2.weight' in weights and weights['classifier.2.weight'].shape[1] == 512:
        return True
    return False


def load_checkpoint_weights(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    return state.get('model', state)


def build_model_for_weights(weights: dict, n_classes: int) -> CRNN:
    legacy = is_legacy_checkpoint(weights)
    legacy_conv = 'cnn.7.block.0.weight' in weights or 'cnn.7.block.1.weight' in weights
    new_conv = 'cnn.8.block.0.weight' in weights or 'cnn.8.block.1.weight' in weights
    use_final_conv = legacy_conv or new_conv
    hidden_size = 256 if legacy else 320
    model = CRNN(n_classes, hidden_size=hidden_size, use_final_conv=use_final_conv, legacy=legacy)
    if legacy:
        log('Legacy checkpoint detected: building compatible architecture')
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='G-VISION OCR trainer/evaluator')
    parser.add_argument('--mode', choices=['train', 'eval', 'infer'], default='train',
                        help='train: continue training; eval: run validation check; infer: recognize one image')
    parser.add_argument('--infer-path', type=str, nargs='+', default=None,
                        help='Path to a single image for inference when mode=infer')
    parser.add_argument('--sample-count', type=int, default=10,
                        help='Number of validation samples to print during eval')
    parser.add_argument('--resume', action='store_true', dest='resume', default=True,
                        help='Resume from checkpoint or best_model.pt if available (default)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Do not resume from checkpoint or best_model.pt; train from scratch')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for training and evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit the number of samples used for training/validation')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of DataLoader workers')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override initial learning rate')
    parser.add_argument('--val-frequency', type=int, default=None,
                        help='Override validation frequency in epochs')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed for dataset sampling and shuffling')
    return parser.parse_args()


def preprocess_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    if img.mode != 'L':
        img = img.convert('L')
    width, height = img.size
    if height != C.IMG_HEIGHT:
        width = max(32, int(round(width * (C.IMG_HEIGHT / height))))
        img = img.resize((width, C.IMG_HEIGHT), Image.Resampling.LANCZOS)
    if img.width > C.MAX_WIDTH:
        img = img.resize((C.MAX_WIDTH, C.IMG_HEIGHT), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(img)
    tensor = 1.0 - tensor
    return tensor.unsqueeze(0).to(device)


def infer_image(image_path: str, model: CRNN, device: torch.device) -> str:
    model.eval()
    img = Image.open(image_path).convert('L')
    x = preprocess_image(img, device)
    with torch.no_grad():
        logits = model(x)
    seq = logits.argmax(2).cpu().numpy().T[0]
    return ctc_greedy_decode(seq)


def evaluate_model(model: CRNN, loader: DataLoader, device: torch.device, max_examples: int = 10) -> tuple[float, list[tuple[str, str]]]:
    model.eval()
    cers = []
    examples = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', ncols=100, leave=False):
            x = batch['imgs'].to(device, non_blocking=True)
            logits = model(x).float()
            preds = logits.argmax(2).cpu().numpy().T
            for seq, target in zip(preds, batch['txts']):
                pred_text = ctc_greedy_decode(seq)
                cers.append(calc_cer(pred_text, target))
                if len(examples) < max_examples:
                    examples.append((target, pred_text))

    return float(np.mean(cers)), examples


def main() -> None:
    args = parse_args()
    if args.infer_path:
        args.infer_path = ' '.join(args.infer_path)
    C.RESUME = args.resume
    if args.epochs is not None:
        C.EPOCHS = args.epochs
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
    if args.max_samples is not None:
        C.MAX_SAMPLES = args.max_samples
    if args.num_workers is not None:
        C.NUM_WORKERS = args.num_workers
    if args.learning_rate is not None:
        C.LR = args.learning_rate
    if args.val_frequency is not None:
        C.VAL_FREQUENCY = args.val_frequency
    if args.seed is not None:
        C.SEED = args.seed

    set_seed(C.SEED)
    log('=' * 60)
    log('G-VISION OCR — Handwritten Optimizer')
    log(f'Mode: {args.mode}')
    log('=' * 60)

    if torch.cuda.is_available():
        log(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        log('CPU-only mode: CUDA not available')

    C.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = C.SAVE_DIR / C.LOG_FILE

    if not C.CACHE_PATH.exists():
        log('Cache file not found. Please run the caching script first.')
        sys.exit(1)

    cached = torch.load(C.CACHE_PATH, weights_only=False)
    all_paths, all_texts = cached['image_paths'], cached['texts']
    dataset_size = len(all_paths)
    log(f'Loaded {dataset_size:,} valid samples')

    if dataset_size > C.MAX_SAMPLES:
        selected = random.sample(range(dataset_size), C.MAX_SAMPLES)
        all_paths = [all_paths[i] for i in selected]
        all_texts = [all_texts[i] for i in selected]
        dataset_size = len(all_paths)
        log(f'Limited to {dataset_size:,} samples for training efficiency')

    indices = list(range(dataset_size))
    random.shuffle(indices)

    n_train = int(dataset_size * 0.90)
    n_val = int(dataset_size * 0.05)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    train_paths = [all_paths[i] for i in train_indices]
    train_texts = [all_texts[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]
    val_texts = [all_texts[i] for i in val_indices]

    train_dataset = OCRDataset(train_paths, train_texts, augment=True)
    val_dataset = OCRDataset(val_paths, val_texts, augment=False)

    device = torch.device(C.DEVICE)
    use_cuda = device.type == 'cuda'
    loader_args = {
        'collate_fn': collate,
        'pin_memory': use_cuda,
        'persistent_workers': C.NUM_WORKERS > 0,
    }
    if C.NUM_WORKERS > 0:
        loader_args['prefetch_factor'] = C.PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=C.NUM_WORKERS,
        drop_last=True,
        **loader_args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
        drop_last=False,
        **loader_args
    )

    start_epoch = 1
    checkpoint_path = C.SAVE_DIR / 'checkpoint.pt'
    best_model_path = C.SAVE_DIR / C.MODEL_NAME

    model = CRNN(VOCAB_SIZE).to(device)
    if C.RESUME and not checkpoint_path.exists() and best_model_path.exists():
        weights = load_checkpoint_weights(best_model_path, device)
        model = build_model_for_weights(weights, VOCAB_SIZE).to(device)

    trainer = Trainer(model, device, len(train_loader))

    if C.RESUME and checkpoint_path.exists():
        loaded_epoch = trainer.load_checkpoint(checkpoint_path, device)
        start_epoch = min(loaded_epoch + 1, C.EPOCHS)
        log(f'Resumed from checkpoint: epoch {loaded_epoch}, best CER={trainer.best_cer:.4f}')
    elif C.RESUME and best_model_path.exists():
        checkpoint = load_model_weights(best_model_path, model, device)
        if isinstance(checkpoint, dict):
            trainer.best_cer = checkpoint.get('metrics', {}).get('cer', trainer.best_cer)
        log(f'Loaded weights from {best_model_path} for fine-tuning')

    if args.mode in ('eval', 'infer'):
        if args.mode == 'eval':
            val_cer, examples = evaluate_model(trainer.model, val_loader, device, args.sample_count)
            log(f'Evaluation complete. CER={val_cer:.4f}')
            for idx, (target, pred) in enumerate(examples, start=1):
                log(f'  {idx}. GT: {target}')
                log(f'     PR: {pred}')
            return

        if args.mode == 'infer':
            if not args.infer_path:
                log('Please specify --infer-path for inference mode')
                sys.exit(1)
            pred_text = infer_image(args.infer_path, trainer.model, device)
            log(f'Inference result: {Path(args.infer_path).name} -> {pred_text}')
            return

    if C.USE_COMPILE and hasattr(torch, 'compile'):
        try:
            import triton  # noqa: F401
            model = torch.compile(model, backend='inductor', mode='reduce-overhead')
            trainer.model = model
            log('torch.compile enabled')
        except ImportError:
            log('Warning: Triton not installed, torch.compile disabled')
        except Exception as exc:
            log(f'Warning: torch.compile failed: {exc}')
            log('Compilation disabled for stability')
            C.USE_COMPILE = False

    log(f'Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}')
    log(f'Starting training: epochs={C.EPOCHS}, effective_batch={C.BATCH_SIZE * C.GRADIENT_ACCUMULATION}, lr={C.LR}, resume_from={start_epoch}')

    for epoch in range(start_epoch, C.EPOCHS + 1):
        epoch_start = time.time()
        train_loss = trainer.train_epoch(train_loader)
        elapsed = time.time() - epoch_start

        if epoch == 1 or epoch % C.VAL_FREQUENCY == 0:
            val_loss, val_cer = trainer.val_epoch(val_loader)
            log(f'Epoch {epoch}/{C.EPOCHS} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | CER={val_cer:.4f} | Time={elapsed/60:.1f}m', log_file)

            if val_cer < trainer.best_cer:
                trainer.best_cer = val_cer
                trainer.patience = 0
                trainer.save(C.SAVE_DIR / C.MODEL_NAME, {'cer': val_cer, 'epoch': epoch}, is_best=True)
                log(f'New best CER: {val_cer:.4f}')
            else:
                trainer.patience += 1
                if trainer.patience >= C.PATIENCE:
                    log(f'Early stopping triggered at epoch {epoch}')
                    break
        else:
            log(f'Epoch {epoch}/{C.EPOCHS} | Train Loss={train_loss:.4f} | Val skipped | Time={elapsed/60:.1f}m', log_file)

        if use_cuda:
            torch.cuda.empty_cache()

    log('=' * 60)
    log(f'Training completed. Best CER: {trainer.best_cer:.4f}')
    log('=' * 60)


if __name__ == '__main__':
    main()
