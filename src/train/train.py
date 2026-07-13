"""
train.py

Faz 2: Pretrained Swin-Base backbone'unu binary classification head ile
fine-tune eden training pipeline'ı.

Özellikler:
- Differential learning rate: backbone için düşük LR, head için yüksek LR
  (ayrı optimizer parameter group'ları ile).
- Early stopping: val F1 (distracted sınıfı) bazlı, `--patience` ile ayarlanır.
- Checkpoint kaydı: sadece en iyi val F1'de, backbone ve head ayrı dosyalara
  (`backbone_best.pth`, `head_best.pth`) — video fazında (Faz 4) backbone
  warm-start için bu ayrım gerekli.

Kullanım (Colab, config dosyasıyla):

    python src/train/train.py --config configs/train_config.yaml

Kullanım (doğrudan CLI argümanlarıyla, config olmadan):

    python src/train/train.py \
        --splits_dir /content/statefarm_splits \
        --images_root /content/statefarm_binary \
        --output_dir /content/checkpoints \
        --epochs 10 --batch_size 32 --lr_backbone 1e-5 --lr_head 1e-3

Not: --config verilirse dosyadaki değerler varsayılan olarak kullanılır,
CLI'da ayrıca verilen argümanlar bu varsayılanları override eder.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import precision_recall_fscore_support

# Bu script doğrudan `python src/train/train.py` ile çalıştırıldığında
# Python repo kökünü sys.path'e otomatik eklemiyor; `src.*` importlarının
# çalışması için repo kökü elle eklenir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_prep.dataset import get_dataloaders, LABEL_TO_IDX  # noqa: E402
from src.model.swin_classifier import SwinDriverClassifier  # noqa: E402


def set_seed(seed: int = 42) -> None:
    """Reproducibility için random seed'leri sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    """Bir epoch boyunca eğitim yapar, ortalama loss ve accuracy döner."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def validate(model, loader, criterion, device) -> dict:
    """
    Val/test set üzerinde metrikleri hesaplar.

    precision/recall/f1, 'distracted' sınıfı pozitif kabul edilerek (binary
    average, pos_label=distracted) hesaplanır — proje önceliği distracted
    örneklerin kaçırılmaması (false negative'lerin azaltılması) olduğu için
    recall değeri doğrudan distracted sınıfının recall'üdür.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        pos_label=LABEL_TO_IDX["distracted"],
        zero_division=0,
    )

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_optimizer(model: SwinDriverClassifier, lr_backbone: float, lr_head: float) -> torch.optim.Optimizer:
    """Differential learning rate: backbone düşük LR, head yüksek LR ile ayrı parameter group'lar."""
    return torch.optim.AdamW([
        {"params": model.get_backbone_parameters(), "lr": lr_backbone},
        {"params": model.get_head_parameters(), "lr": lr_head},
    ])


def save_checkpoints(model: SwinDriverClassifier, output_dir: Path) -> None:
    """Backbone ve head ağırlıklarını ayrı dosyalara kaydeder (Faz 4 warm-start için)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.get_backbone_state_dict(), output_dir / "backbone_best.pth")
    torch.save(model.get_head_state_dict(), output_dir / "head_best.pth")


def parse_args() -> argparse.Namespace:
    # İlk geçiş: sadece --config okunur, böylece dosyadaki değerler ana
    # parser'ın varsayılanları olarak kullanılabilir (CLI argümanları bu
    # varsayılanları her zaman override eder).
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    file_defaults = {}
    if config_args.config:
        config_path = Path(config_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            file_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(
        description="Swin-Base binary classifier'ı fine-tune eder.",
        parents=[config_parser],
    )
    parser.add_argument("--splits_dir", type=str, default=file_defaults.get("splits_dir"))
    parser.add_argument("--images_root", type=str, default=file_defaults.get("images_root"))
    parser.add_argument("--output_dir", type=str, default=file_defaults.get("output_dir"))
    parser.add_argument("--epochs", type=int, default=file_defaults.get("epochs", 10))
    parser.add_argument("--batch_size", type=int, default=file_defaults.get("batch_size", 32))
    parser.add_argument("--lr_backbone", type=float, default=file_defaults.get("lr_backbone", 1e-5))
    parser.add_argument("--lr_head", type=float, default=file_defaults.get("lr_head", 1e-3))
    parser.add_argument("--patience", type=int, default=file_defaults.get("patience", 3))
    parser.add_argument("--num_workers", type=int, default=file_defaults.get("num_workers", 4))
    parser.add_argument("--seed", type=int, default=file_defaults.get("seed", 42))
    args = parser.parse_args()

    missing = [name for name in ("splits_dir", "images_root", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error(
            f"Şu argümanlar zorunlu: {missing} — CLI'da doğrudan ya da --config "
            f"dosyası içinde verilmeli."
        )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    loaders = get_dataloaders(
        splits_dir=args.splits_dir,
        images_root=args.images_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SwinDriverClassifier(num_classes=2).to(device)
    optimizer = build_optimizer(model, args.lr_backbone, args.lr_head)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = validate(model, loaders["val"], criterion, device)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_precision(distracted)={val_metrics['precision']:.4f} "
            f"val_recall(distracted)={val_metrics['recall']:.4f} "
            f"val_f1(distracted)={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoints(model, output_dir)
            print(f"  -> Yeni en iyi val F1 (distracted): {best_f1:.4f}, checkpoint kaydedildi: {output_dir}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: {args.patience} epoch boyunca val F1 iyileşmedi.")
                break

    print("\n--- Eğitim Tamamlandı ---")
    print(f"En iyi val F1 (distracted): {best_f1:.4f}")
    print(f"Checkpoint'ler: {output_dir / 'backbone_best.pth'}, {output_dir / 'head_best.pth'}")


if __name__ == "__main__":
    main()
