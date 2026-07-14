"""
prepare_external_dataset.py

Roboflow'dan (veya benzer c0-c9 klasör isimlendirmesine sahip bir kaynaktan)
indirilen external driver-distraction veri setini, State Farm ile aynı
ikili (attentive/distracted) formata çevirir. Çıktısı, evaluate.py'nin
--external_splits_dir / --external_images_root argümanlarıyla doğrudan
kullanılabilir.

Kaynak klasör yapısı (Roboflow 'Folder Structure' export'u ya da benzeri):

    source_dir/
        train/c0../*.jpg
        valid/  (veya "val") /c0../*.jpg
        test/c0../*.jpg

Alt küme (subset) klasör isimleri kaynağa göre değişebilir (Roboflow
"valid" kullanırken bazı kaynaklar "val" kullanıyor) -- script, source_dir
altında hangi aday isimler gerçekten mevcutsa onları otomatik kullanır.
Sınıf klasörleri de her zaman c0-c9'un tamamını içermek zorunda değildir
(örn. sadece c0-c3 olabilir); CLASS_MAP'te olup kaynakta bulunmayan
sınıflar uyarı ile atlanır, hata vermez.

Bu script train/val/test ayrımını yok sayar -- bu veri modele hiçbir
aşamada (ne eğitimde ne model seçiminde) gösterilmeyecek, sadece final
external değerlendirme için kullanılacağından, hepsi TEK bir external
test setinde birleştirilir. Bu yüzden subject_split.py'deki gibi
sürücü bazlı ayrıma da gerek yoktur.

Kullanım:

    python src/data_prep/prepare_external_dataset.py \
        --source /content/external_raw \
        --images_root /content/external_binary \
        --splits_dir /content/external_splits
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


# State Farm ile aynı sınıf isimlendirmesi (c0-c9) kullanıldığı için
# binary_mapping.py ile aynı eşleme.
CLASS_MAP = {
    "c0": "attentive",   # safe driving
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

# Kaynağa göre değişen olası alt küme (subset) klasör isimleri.
# source_dir altında hangileri gerçekten mevcutsa sadece onlar kullanılır.
SOURCE_SUBSET_CANDIDATES = ["train", "valid", "val", "test"]


def build_external_dataset(source_dir: Path, images_root: Path, splits_dir: Path, image_ext: str = "*.jpg") -> None:
    """
    source_dir altındaki (train/valid veya val/test)/c0.. klasörlerini
    okuyup images_root altında attentive/distracted klasörlerine kopyalar,
    tek bir splits_dir/test.csv üretir.
    """
    source_dir = Path(source_dir)
    images_root = Path(images_root)
    splits_dir = Path(splits_dir)

    subsets_found = [s for s in SOURCE_SUBSET_CANDIDATES if (source_dir / s).is_dir()]
    if not subsets_found:
        raise FileNotFoundError(
            f"{source_dir} altında beklenen alt küme klasörlerinden ({SOURCE_SUBSET_CANDIDATES}) "
            f"hiçbiri bulunamadı. Kaynak yolunu kontrol et."
        )
    print(f"Bulunan alt küme klasörleri: {subsets_found}")

    for binary_class in set(CLASS_MAP.values()):
        (images_root / binary_class).mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {binary_class: 0 for binary_class in set(CLASS_MAP.values())}

    for subset in subsets_found:
        for original_class, binary_class in CLASS_MAP.items():
            src = source_dir / subset / original_class
            if not src.exists():
                print(f"[UYARI] Kaynak klasör bulunamadı, atlanıyor: {src}")
                continue

            for img_path in src.glob(image_ext):
                # Farklı subset'lerden (train/valid/test) gelen aynı isimli
                # dosyaların birbirini ezmemesi için subset adı da dosya
                # adına eklenir.
                img_name = f"{subset}_{img_path.name}"
                new_name = f"{original_class}_{img_name}"
                shutil.copy(img_path, images_root / binary_class / new_name)
                rows.append({
                    "classname": original_class,
                    "img": img_name,
                    "binary_label": binary_class,
                    "source_subset": subset,
                })
                counts[binary_class] += 1

    df = pd.DataFrame(rows)
    df.to_csv(splits_dir / "test.csv", index=False)

    print("\n--- External Dataset Hazırlığı Tamamlandı ---")
    for binary_class, count in counts.items():
        print(f"{binary_class}: {count} görsel")
    total = sum(counts.values())
    if total > 0:
        for binary_class, count in counts.items():
            print(f"  {binary_class} oranı: {count / total:.2%}")
    else:
        print("[UYARI] Hiç görsel bulunamadı, kaynak klasör yapısını kontrol et.")
    print(f"Yazıldı: {splits_dir / 'test.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External driver-distraction veri setini ikili formata çevirir.")
    parser.add_argument(
        "--source", type=str, required=True,
        help="External veri setinin kök klasörü (train/valid-veya-val/test alt klasörlerini, her biri c0.. sınıf klasörlerini içerir).",
    )
    parser.add_argument(
        "--images_root", type=str, required=True,
        help="attentive/distracted klasörlerinin yazılacağı hedef klasör.",
    )
    parser.add_argument(
        "--splits_dir", type=str, required=True,
        help="test.csv'nin yazılacağı klasör.",
    )
    parser.add_argument("--ext", type=str, default="*.jpg", help="İşlenecek görsel uzantısı (varsayılan: *.jpg).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_external_dataset(
        source_dir=Path(args.source),
        images_root=Path(args.images_root),
        splits_dir=Path(args.splits_dir),
        image_ext=args.ext,
    )
