"""Train a FaceNet-based 5-class FER classifier.

Classes:
    anger, disgust, happy, neutral, surprise

The source datasets use the common 7-class FER label IDs:
    anger=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6

This script excludes fear and sad, then remaps the remaining labels to 0..4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from datasets import load_dataset
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset


CLASSES = ["anger", "disgust", "happy", "neutral", "surprise"]
KEEP_LABELS = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FaceNet 5-class FER")
    parser.add_argument("--train-dataset", default="Piro17/dataset-affecthqnet-fer2013")
    parser.add_argument("--test-dataset", default="Piro17/fer2013test")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offline-datasets",
        action="store_true",
        help="Use cached Hugging Face datasets only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class FaceNetFER(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = InceptionResnetV1(pretrained="vggface2")
        backbone.classify = False
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class FERDataset5(Dataset):
    def __init__(self, hf_ds, transform=None) -> None:
        all_labels = hf_ds["label"]
        self.samples = [
            (idx, KEEP_LABELS[int(label)])
            for idx, label in enumerate(all_labels)
            if int(label) in KEEP_LABELS
        ]
        self.data = hf_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        original_index, label = self.samples[index]
        image = self.data[original_index]["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split: str = "train") -> T.Compose:
    mean, std = [0.5] * 3, [0.5] * 3
    if split == "train":
        return T.Compose(
            [
                T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                T.RandomGrayscale(0.05),
                T.ToTensor(),
                T.Normalize(mean, std),
                T.RandomErasing(0.2, scale=(0.02, 0.12)),
            ]
        )
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(mean, std)])


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        predictions.extend(logits.argmax(1).cpu().numpy())
        targets.extend(labels.numpy())
    accuracy = sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)
    return accuracy, predictions, targets


def make_loaders(args: argparse.Namespace):
    raw_train = load_dataset(args.train_dataset, split="train")
    raw_test = load_dataset(args.test_dataset, split="train")
    split = raw_train.train_test_split(test_size=0.1, seed=args.seed)

    train_ds = FERDataset5(split["train"], get_transforms("train"))
    val_ds = FERDataset5(split["test"], get_transforms("val"))
    test_ds = FERDataset5(raw_test, get_transforms("test"))

    dist = Counter(label for _, label in train_ds.samples)
    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"class distribution: {{ {', '.join(f'{CLASSES[k]}={v}' for k, v in sorted(dist.items()))} }}")

    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def run() -> None:
    args = parse_args()
    if args.offline_datasets:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "facenet_5class_best.pth"
    result_path = args.output_dir / "result.json"
    swa_start = int(args.epochs * 0.75)

    print("=" * 60)
    print("FaceNet-VGGFace2 5-class FER")
    print(f"classes: {CLASSES}")
    print("excluded classes: fear, sad")
    print(f"device={device}, batch_size={args.batch_size}")
    print("=" * 60)

    train_loader, val_loader, test_loader = make_loaders(args)

    model = FaceNetFER().to(device)
    print(f"params: {sum(param.numel() for param in model.parameters()) / 1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.freeze_backbone()
    optimizer_head = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr * 5,
    )
    print(f"[stage 1] freeze backbone for {args.freeze_epochs} epochs")
    for epoch in range(1, args.freeze_epochs + 1):
        started_at = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_head, scaler, device
        )
        val_acc, _, _ = evaluate(model, val_loader, device)
        memory_gb = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(
            f"  [{epoch}/{args.freeze_epochs}] "
            f"loss={train_loss:.4f} train={train_acc * 100:.2f}% "
            f"val={val_acc * 100:.2f}% {time.time() - started_at:.0f}s "
            f"vram={memory_gb:.1f}GB"
        )
        sys.stdout.flush()

    model.unfreeze_all()
    print("[stage 1] done")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.02)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_acc, _, _ = evaluate(model, val_loader, device)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        memory_gb = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"loss={train_loss:.4f} train={train_acc * 100:.2f}% "
            f"val={val_acc * 100:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"{time.time() - started_at:.0f}s vram={memory_gb:.1f}GB"
        )
        sys.stdout.flush()

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    print("updating SWA batch norm...")
    update_bn(train_loader, swa_model, device=device)
    swa_val, _, _ = evaluate(swa_model, val_loader, device)
    print(f"swa_val={swa_val * 100:.2f}% best_val={best_val * 100:.2f}%")
    eval_model = model
    if swa_val > best_val:
        best_val = swa_val
        torch.save(swa_model.module.state_dict(), best_path)
        eval_model = swa_model.module

    eval_model.load_state_dict(load_state_dict(best_path, device))
    test_acc, predictions, targets = evaluate(eval_model, test_loader, device)

    print("=" * 50)
    print(f"TEST ACCURACY: {test_acc * 100:.2f}%")
    print("=" * 50)
    matrix = confusion_matrix(targets, predictions, labels=list(range(NUM_CLASSES)))
    per_class = np.divide(
        matrix.diagonal(),
        matrix.sum(axis=1),
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=matrix.sum(axis=1) != 0,
    )
    for index, class_name in enumerate(CLASSES):
        print(f"{class_name:10s}: {per_class[index] * 100:5.1f}%")
    print()
    print(classification_report(targets, predictions, target_names=CLASSES, digits=3))

    result = {
        "test_acc": round(test_acc * 100, 2),
        "best_val": round(best_val * 100, 2),
        "classes": CLASSES,
        "per_class": {
            CLASSES[index]: round(float(per_class[index]) * 100, 1)
            for index in range(NUM_CLASSES)
        },
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done. checkpoint={best_path}")


if __name__ == "__main__":
    run()
