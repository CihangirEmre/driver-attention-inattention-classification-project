"""
evaluate.py

Faz 3: Eğitilmiş modelin (backbone_best.pth + head_best.pth) internal
(State Farm test split) ve external (örn. AUC Distracted Driver Dataset)
performansını ölçer.

Internal ve external değerlendirme aynı `run_evaluation` fonksiyonunu
kullanır (kod tekrarı yok). --splits_dir/--images_root (internal) ve
--external_splits_dir/--external_images_root (external) birbirinden
bağımsız opsiyoneldir; en az biri verilmelidir. İkisi de verilirse
aralarındaki fark "generalization gap" olarak ayrıca raporlanır.

Kullanım (sadece internal):

    python src/eval/evaluate.py \
        --checkpoint_backbone /content/checkpoints/backbone_best.pth \
        --checkpoint_head /content/checkpoints/head_best.pth \
        --splits_dir /content/statefarm_splits \
        --images_root /content/statefarm_binary \
        --output_json /content/eval_results.json

Kullanım (sadece external):

    python src/eval/evaluate.py \
        --checkpoint_backbone /content/checkpoints/backbone_best.pth \
        --checkpoint_head /content/checkpoints/head_best.pth \
        --external_splits_dir /content/external_splits \
        --external_images_root /content/external_binary \
        --output_json /content/eval_results.json

Kullanım (ikisi birlikte, generalization gap raporlanır):

    python src/eval/evaluate.py \
        --checkpoint_backbone /content/checkpoints/backbone_best.pth \
        --checkpoint_head /content/checkpoints/head_best.pth \
        --splits_dir /content/statefarm_splits \
        --images_root /content/statefarm_binary \
        --external_splits_dir /content/external_splits \
        --external_images_root /content/external_binary \
        --output_json /content/eval_results.json

Not: external veri de subject_split.py/binary_mapping.py ile üretilenle
aynı formatta (images_root/{attentive,distracted}/{classname}_{img},
splits_dir/test.csv) hazırlanmış olmalıdır.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_prep.dataset import DriverAttentionDataset, LABEL_TO_IDX, build_transforms, get_dataloaders  # noqa: E402
from src.model.swin_classifier import SwinDriverClassifier  # noqa: E402


def load_checkpoint(model: SwinDriverClassifier, backbone_path: str, head_path: str, device: str) -> SwinDriverClassifier:
    """train.py'nin ayrı kaydettiği backbone/head state_dict'lerini modele yükler."""
    backbone_sd = torch.load(backbone_path, map_location=device, weights_only=True)
    head_sd = torch.load(head_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict({**backbone_sd, **head_sd})
    return model


@torch.no_grad()
def run_evaluation(model: SwinDriverClassifier, loader: DataLoader, device: str) -> dict:
    """
    Verilen loader üzerinde modeli değerlendirir.

    precision/recall/f1, train.py'deki validate ile tutarlı şekilde
    'distracted' sınıfı pozitif kabul edilerek (binary average) hesaplanır.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        pos_label=LABEL_TO_IDX["distracted"],
        zero_division=0,
    )

    return {
        "n_samples": len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist(),
    }


def evaluate_external(
    model: SwinDriverClassifier,
    external_splits_dir: str,
    external_images_root: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """
    External dataset üzerinde run_evaluation ile aynı metrikleri hesaplar.

    get_dataloaders() train/val/test üçlüsünü zorunlu kıldığı için (external
    veride bu üçlü olmayabilir), external_splits_dir altında sadece
    'test.csv' bekleyip DriverAttentionDataset'i doğrudan kullanır.
    """
    csv_path = Path(external_splits_dir) / "test.csv"
    transform = build_transforms(image_size=224)["test"]
    dataset = DriverAttentionDataset(csv_path=csv_path, images_root=external_images_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return run_evaluation(model, loader, device)


def print_metrics(metrics: dict) -> None:
    print(f"  n_samples             : {metrics['n_samples']}")
    print(f"  accuracy               : {metrics['accuracy']:.4f}")
    print(f"  precision (distracted) : {metrics['precision']:.4f}")
    print(f"  recall (distracted)    : {metrics['recall']:.4f}")
    print(f"  f1 (distracted)        : {metrics['f1']:.4f}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]] (0=attentive,1=distracted): {metrics['confusion_matrix']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eğitilmiş modelin internal/external performansını değerlendirir.")
    parser.add_argument("--checkpoint_backbone", type=str, required=True)
    parser.add_argument("--checkpoint_head", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default=None, help="Opsiyonel: Internal (State Farm) splits klasörü.")
    parser.add_argument("--images_root", type=str, default=None, help="Opsiyonel: Internal (State Farm) images klasörü.")
    parser.add_argument("--external_splits_dir", type=str, default=None, help="Opsiyonel: external dataset splits klasörü.")
    parser.add_argument("--external_images_root", type=str, default=None, help="Opsiyonel: external dataset images klasörü.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_json", type=str, default=None, help="Sonuçların yazılacağı JSON dosyası (opsiyonel).")
    args = parser.parse_args()

    if bool(args.splits_dir) != bool(args.images_root):
        parser.error("--splits_dir ve --images_root birlikte verilmeli.")
    if bool(args.external_splits_dir) != bool(args.external_images_root):
        parser.error("--external_splits_dir ve --external_images_root birlikte verilmeli.")
    if not args.splits_dir and not args.external_splits_dir:
        parser.error("En az biri verilmeli: internal (--splits_dir/--images_root) ya da external (--external_splits_dir/--external_images_root).")

    return args


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    model = SwinDriverClassifier(pretrained=False, num_classes=2).to(device)
    model = load_checkpoint(model, args.checkpoint_backbone, args.checkpoint_head, device)

    results = {}
    internal_metrics = None

    if args.splits_dir:
        loaders = get_dataloaders(
            splits_dir=args.splits_dir,
            images_root=args.images_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_weighted_sampler=False,
        )

        internal_metrics = run_evaluation(model, loaders["test"], device)
        print("\n--- Internal (State Farm test split) Sonuçları ---")
        print_metrics(internal_metrics)
        results["internal"] = internal_metrics

    if args.external_splits_dir:
        external_metrics = evaluate_external(
            model,
            args.external_splits_dir,
            args.external_images_root,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("\n--- External Dataset Sonuçları ---")
        print_metrics(external_metrics)
        results["external"] = external_metrics

        if internal_metrics is not None:
            gap = {
                "accuracy_gap": internal_metrics["accuracy"] - external_metrics["accuracy"],
                "recall_gap": internal_metrics["recall"] - external_metrics["recall"],
                "f1_gap": internal_metrics["f1"] - external_metrics["f1"],
            }
            print("\n--- Generalization Gap (internal - external) ---")
            for key, value in gap.items():
                print(f"  {key}: {value:+.4f}")
            results["generalization_gap"] = gap

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSonuçlar JSON'a yazıldı: {output_path}")


if __name__ == "__main__":
    main()
