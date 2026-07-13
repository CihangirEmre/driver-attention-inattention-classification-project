"""
dataset.py

subject_split.py'nin ürettiği split CSV'lerini (train.csv/val.csv/test.csv)
ve binary_mapping.py'nin ürettiği fiziksel görsel klasörünü (attentive/distracted)
birleştirerek PyTorch Dataset ve DataLoader'ları oluşturur.

Beklenen klasör/dosya yapısı:

    images_root/
        attentive/
            c0_img_1.jpg
            c0_img_2.jpg
            ...
        distracted/
            c1_img_1.jpg
            c5_img_7.jpg
            ...

    splits_dir/
        train.csv   (kolonlar: subject, classname, img, binary_label, split)
        val.csv
        test.csv

Dosya eşleştirme mantığı: binary_mapping.py her görseli
f"{classname}_{orijinal_dosya_adı}" olarak kopyaladığı için,
split CSV'sindeki (classname, img) çiftinden gerçek dosya yolu yeniden inşa edilir.

Kullanım (Colab içinde):

    from src.data_prep.dataset import get_dataloaders

    loaders = get_dataloaders(
        splits_dir="/content/statefarm_splits",
        images_root="/content/statefarm_binary",
        batch_size=32,
        image_size=224,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


LABEL_TO_IDX = {"attentive": 0, "distracted": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# Swin-Base (timm) varsayılan normalizasyon değerleri (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DriverAttentionDataset(Dataset):
    """
    split CSV'sinden ve images_root'tan görselleri okuyan Dataset sınıfı.
    """

    def __init__(self, csv_path: Path, images_root: Path, transform: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.transform = transform

        missing_labels = set(self.df["binary_label"].unique()) - set(LABEL_TO_IDX.keys())
        if missing_labels:
            raise ValueError(f"Beklenmeyen binary_label değer(ler)i: {missing_labels}")

        # Dosya var mı diye erken kontrol (sessiz hatalardan kaçınmak için)
        self._validate_first_n(n=5)

    def _build_path(self, row: pd.Series) -> Path:
        filename = f"{row['classname']}_{row['img']}"
        return self.images_root / row["binary_label"] / filename

    def _validate_first_n(self, n: int = 5) -> None:
        for i in range(min(n, len(self.df))):
            row = self.df.iloc[i]
            path = self._build_path(row)
            if not path.exists():
                raise FileNotFoundError(
                    f"Beklenen görsel bulunamadı: {path}\n"
                    f"binary_mapping.py'nin bu split'ten önce çalıştırıldığından "
                    f"ve images_root yolunun doğru olduğundan emin ol."
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._build_path(row)

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = LABEL_TO_IDX[row["binary_label"]]
        return image, label


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Train ve val/test için ayrı transform pipeline'ları döner.
    Val/test'te augmentation yok, sadece resize + normalize (Faz 1.4'te
    sabitlenen preprocessing pipeline'ı ile tutarlı).
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def build_weighted_sampler(dataset: DriverAttentionDataset) -> WeightedRandomSampler:
    """
    Class imbalance için WeightedRandomSampler oluşturur (sadece train split'inde kullanılır).
    Distracted sınıfı doğal olarak baskın olduğu için (9 alt-davranıştan geldiği için),
    her örneğe kendi sınıfının ters frekansıyla orantılı ağırlık verilir.
    """
    labels = dataset.df["binary_label"].map(LABEL_TO_IDX).values
    class_counts = pd.Series(labels).value_counts().sort_index()
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    splits_dir: str,
    images_root: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    train/val/test CSV'lerini okuyup ilgili DataLoader'ları döner.
    """
    splits_dir = Path(splits_dir)
    images_root = Path(images_root)
    transform_map = build_transforms(image_size=image_size)

    datasets = {
        split: DriverAttentionDataset(
            csv_path=splits_dir / f"{split}.csv",
            images_root=images_root,
            transform=transform_map[split],
        )
        for split in ["train", "val", "test"]
    }

    loaders = {}

    # Train: WeightedRandomSampler kullanılıyorsa shuffle=False olmalı (sampler zaten örnekliyor)
    if use_weighted_sampler:
        sampler = build_weighted_sampler(datasets["train"])
        loaders["train"] = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loaders["train"] = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    for split in ["val", "test"]:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


if __name__ == "__main__":
    # Hızlı manuel test / sanity check için
    import argparse

    parser = argparse.ArgumentParser(description="DataLoader'ları test et.")
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    loaders = get_dataloaders(
        splits_dir=args.splits_dir,
        images_root=args.images_root,
        batch_size=args.batch_size,
    )

    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"{split}: batch şekli={tuple(images.shape)}, label örnekleri={labels[:8].tolist()}")
