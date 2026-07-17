"""
binary_mapping.py

State Farm Distracted Driver Detection veri setindeki c0-c9 sınıflarını
ikili (attentive / distracted) sınıfa indirger.

Kullanım (local veya Colab'da aynı şekilde çalışır):

    python binary_mapping.py \
        --source /content/statefarm/imgs/train \
        --target /content/statefarm_binary

- c0 = safe driving -> attentive
  c1-c9 = distracted alt-davranışları -> distracted
"""

import argparse
import shutil
from pathlib import Path


# State Farm orijinal sınıf -> ikili sınıf eşlemesi
CLASS_MAP = {
    "c0": "attentive",   # safe driving
    "c1": "distracted",  # texting - right
    "c2": "distracted",  # talking on the phone - right
    "c3": "distracted",  # texting - left
    "c4": "distracted",  # talking on the phone - left
    "c5": "distracted",  # operating the radio
    "c6": "distracted",  # drinking
    "c7": "distracted",  # reaching behind
    "c8": "distracted",  # hair and makeup
    "c9": "distracted",  # talking to passenger
}


def build_binary_dataset(source_dir: Path, target_dir: Path, image_ext: str = "*.jpg") -> None:
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    for binary_class in set(CLASS_MAP.values()):
        (target_dir / binary_class).mkdir(parents=True, exist_ok=True)

    counts = {binary_class: 0 for binary_class in set(CLASS_MAP.values())}

    for original_class, binary_class in CLASS_MAP.items():
        src = source_dir / original_class
        if not src.exists():
            print(f"[UYARI] Kaynak klasör bulunamadı, atlanıyor: {src}")
            continue

        dst = target_dir / binary_class
        for img_path in src.glob(image_ext):
            new_name = f"{original_class}_{img_path.name}"
            shutil.copy(img_path, dst / new_name)
            counts[binary_class] += 1

    print("\n--- Binary Mapping Tamamlandı ---")
    for binary_class, count in counts.items():
        print(f"{binary_class}: {count} görsel")
    total = sum(counts.values())
    if total > 0:
        for binary_class, count in counts.items():
            print(f"  {binary_class} oranı: {count / total:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State Farm verisini ikili sınıfa indirger.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Orijinal State Farm train klasörü (c0..c9 alt klasörlerini içerir).",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="İkili sınıflandırılmış verinin yazılacağı klasör.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="*.jpg",
        help="İşlenecek görsel uzantısı (varsayılan: *.jpg).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_binary_dataset(Path(args.source), Path(args.target), args.ext)
