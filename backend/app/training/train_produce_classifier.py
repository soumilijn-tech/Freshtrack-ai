"""Produce Classifier Trainer and ONNX Exporter for FreshTrack AI.
Fine-tunes MobileNetV2 on broad produce classes (including Persimmon vs Orange).
Evaluates test metrics (Precision, Recall, F1, Confusion Matrix) and exports to ONNX.
"""
import os
import sys

# Force UTF-8 encoding and unbuffered output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "produce_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_EXPORT_PATH = MODELS_DIR / "freshtrack_produce_v2.onnx"
CLASSES_EXPORT_PATH = MODELS_DIR / "produce_classes.json"


class ProduceDataset(Dataset):
    """PyTorch Dataset loading images from manifest or folder."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of (Path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                if self.transform:
                    im = self.transform(im)
                return im, label
        except Exception as e:
            # Fallback for corrupted image: return blank image
            blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank, label


def get_transforms():
    """Produce training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_dataset_splits(min_samples_per_class: int = 5):
    """Load train, val, and test splits from DATASET_DIR filtering under-represented classes."""
    manifest_path = DATASET_DIR / "splits.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Only retain classes with at least min_samples_per_class
        classes = sorted([
            c for c, count in manifest["counts_per_class"].items()
            if count >= min_samples_per_class
        ])
        class_to_idx = {c: i for i, c in enumerate(classes)}

        train_samples = [(PROJECT_ROOT / p, class_to_idx[c]) for p, c in manifest["splits"]["train"] if c in class_to_idx]
        val_samples = [(PROJECT_ROOT / p, class_to_idx[c]) for p, c in manifest["splits"]["val"] if c in class_to_idx]
        test_samples = [(PROJECT_ROOT / p, class_to_idx[c]) for p, c in manifest["splits"]["test"] if c in class_to_idx]
    else:
        # Fallback: discover directly from category folders
        classes = sorted([
            d.name for d in DATASET_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".") and len(list(d.glob("*.jpg"))) >= min_samples_per_class
        ])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        import random
        random.seed(42)
        train_samples, val_samples, test_samples = [], [], []

        for c in classes:
            c_dir = DATASET_DIR / c
            files = sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")))
            random.shuffle(files)
            n = len(files)
            n_tr = max(int(n * 0.70), 1)
            n_va = max(int(n * 0.15), 1)
            train_samples.extend([(f, class_to_idx[c]) for f in files[:n_tr]])
            val_samples.extend([(f, class_to_idx[c]) for f in files[n_tr:n_tr + n_va]])
            test_samples.extend([(f, class_to_idx[c]) for f in files[n_tr + n_va:] if n_tr + n_va < n] or [(files[-1], class_to_idx[c])])

    return classes, class_to_idx, train_samples, val_samples, test_samples


def build_model(num_classes: int):
    """Build MobileNetV2 with custom classifier head."""
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze earlier feature layers, fine-tune later layers
    for param in model.features[:12].parameters():
        param.requires_grad = False
    for param in model.features[12:].parameters():
        param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_model(epochs=12, batch_size=16, lr=1e-3):
    """Train MobileNetV2 on produce dataset and export to ONNX."""
    print("=" * 65)
    print("FreshTrack AI — Produce Classifier Fine-Tuning")
    print("=" * 65)

    classes, class_to_idx, train_samples, val_samples, test_samples = load_dataset_splits()
    print(f"Loaded {len(classes)} classes: {classes}")
    print(f"Dataset counts: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    train_trans, val_trans = get_transforms()
    train_loader = DataLoader(ProduceDataset(train_samples, train_trans), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ProduceDataset(val_samples, val_trans), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ProduceDataset(test_samples, val_trans), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = build_model(len(classes)).to(device)

    # Optimizer with differential learning rates
    optimizer = optim.AdamW([
        {"params": model.features[12:].parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr}
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_weights = None

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_acc = train_correct / len(train_samples)
        avg_train_loss = train_loss / len(train_samples)

        # Validation Phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_samples)
        avg_val_loss = val_loss / len(val_samples)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:5.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:5.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc*100:.1f}%")

    if best_weights:
        model.load_state_dict(best_weights)

    # Test Evaluation
    print("\n" + "=" * 65)
    print("Evaluating Model on Strictly Unseen Test Set")
    print("=" * 65)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification Report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=3,
        zero_division=0
    )
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print("Classes:", {i: c for i, c in enumerate(classes)})
    print(cm)

    # Persimmon vs Orange validation
    if "Persimmon" in class_to_idx and "Orange" in class_to_idx:
        p_idx = class_to_idx["Persimmon"]
        o_idx = class_to_idx["Orange"]
        print(f"\nDistinction Check: Persimmon (idx {p_idx}) vs Orange (idx {o_idx}):")
        print(f"  Persimmon classified as Orange: {cm[p_idx, o_idx]}")
        print(f"  Orange classified as Persimmon: {cm[o_idx, p_idx]}")
        p_acc = cm[p_idx, p_idx] / max(np.sum(cm[p_idx, :]), 1) * 100
        o_acc = cm[o_idx, o_idx] / max(np.sum(cm[o_idx, :]), 1) * 100
        print(f"  Persimmon Recall: {p_acc:.1f}%")
        print(f"  Orange Recall:    {o_acc:.1f}%")

    # Save PyTorch weights checkpoint
    torch_checkpoint_path = MODELS_DIR / "freshtrack_produce_v2.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "best_val_acc": best_val_acc,
    }, str(torch_checkpoint_path))
    print(f"Saved PyTorch weights checkpoint to: {torch_checkpoint_path}")

    # Export to ONNX
    print("\n" + "=" * 65)
    print(f"Exporting ONNX model to: {ONNX_EXPORT_PATH}")
    print("=" * 65)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_EXPORT_PATH),
        export_params=True,
        opset_version=14,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    onnx_size_mb = ONNX_EXPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX file size: {onnx_size_mb:.2f} MB")

    # Save classes mapping
    with open(CLASSES_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "FreshTrack Produce Classifier v2",
            "architecture": "MobileNetV2 (Transfer Learned)",
            "num_classes": len(classes),
            "classes": classes,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_shape": [1, 3, 224, 224],
            "confidence_thresholds": {
                "ambiguous_cutoff": 0.35,
                "high_confidence_cutoff": 0.70
            }
        }, f, indent=2)
    print(f"Saved produce classes to {CLASSES_EXPORT_PATH}")

    return {
        "classes": classes,
        "onnx_path": str(ONNX_EXPORT_PATH),
        "onnx_size_mb": onnx_size_mb,
        "best_val_acc": best_val_acc
    }


if __name__ == "__main__":
    train_model(epochs=12, batch_size=16, lr=1e-3)
