"""
subject_split.py

State Farm'ın driver_imgs_list.csv dosyasını kullanarak, sürücü (subject)
bazlı train/val/test split'i oluşturur. Bu kritik bir adımdır: aynı sürücünün
görüntüleri farklı split'lere dağılırsa model gerçekte genelleme yapmadan
yüksek accuracy gösterebilir (data leakage).

Kullanım:

    python subject_split.py \
        --csv /content/statefarm/driver_imgs_list.csv \
        --output_dir /content/statefarm_splits \
        --test_size 0.15 \
        --val_size 0.15

Çıktı:
    output_dir/train.csv
    output_dir/val.csv
    output_dir/test.csv

Her CSV şu kolonları içerir: subject, classname, img, binary_label, split
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CLASS_MAP = {
    "c0": "attentive",
    "c1": "distracted",
    "c2": "distracted",
    "c3": "distracted",
    "c4": "distracted",
    "c5": "distracted",
    "c6": "distracted",
    "c7": "distracted",
    "c8": "distracted",
    "c9": "distracted",
}


def assert_no_leakage(*dfs: pd.DataFrame) -> None:
    """Split'ler arasında ortak subject olmadığını doğrular."""
    subject_sets = [set(df["subject"]) for df in dfs]
    for i in range(len(subject_sets)):
        for j in range(i + 1, len(subject_sets)):
            overlap = subject_sets[i] & subject_sets[j]
            assert not overlap, f"Subject sızıntısı tespit edildi: {overlap}"


def subject_based_split(
    csv_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    driver_imgs_list.csv'yi okuyup subject bazlı train/val/test split'i döner.
    """
    df = pd.read_csv(csv_path)
    df["binary_label"] = df["classname"].map(CLASS_MAP)

    if df["binary_label"].isna().any():
        missing = df[df["binary_label"].isna()]["classname"].unique()
        raise ValueError(f"CLASS_MAP içinde tanımsız sınıf(lar) bulundu: {missing}")

    # 1. adım: test'i ayır (subject bazlı)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["subject"]))
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # 2. adım: kalan train_val'dan val'i ayır (yine subject bazlı)
    # val_size, orijinal veri setine oranla verildiği için train_val içindeki
    # göreli oranı ayarlıyoruz.
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["subject"]))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    assert_no_leakage(train_df, val_df, test_df)

    return {"train": train_df, "val": val_df, "test": test_df}


def print_summary(splits: dict) -> None:
    print("\n--- Subject-Based Split Özeti ---")
    for split_name, split_df in splits.items():
        n_subjects = split_df["subject"].nunique()
        n_images = len(split_df)
        class_dist = split_df["binary_label"].value_counts(normalize=True).to_dict()
        print(f"\n{split_name.upper()}")
        print(f"  Görsel sayısı : {n_images}")
        print(f"  Sürücü sayısı : {n_subjects}")
        print(f"  Sınıf dağılımı: {class_dist}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sürücü bazlı train/val/test split oluşturur.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="State Farm driver_imgs_list.csv yolu.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Split CSV'lerinin yazılacağı klasör.",
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    splits = subject_based_split(
        csv_path=Path(args.csv),
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Yazıldı: {out_path}")

    print_summary(splits)
