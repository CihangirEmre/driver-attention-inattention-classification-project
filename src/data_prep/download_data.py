"""
download_data.py

State Farm Distracted Driver Detection veri setini Kaggle API üzerinden
indirir ve açar. Colab'da (veya local'de) çalıştırılabilir.

Ön koşullar:
1. Kaggle hesabından API token indirilmiş olmalı (kaggle.json)
   Kaggle -> Account -> API -> Create New Token
2. https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules
   sayfasından "Join Competition" ile competition kurallarını kabul etmiş olman gerekir
   (aksi halde indirme 403 hatası verir).
3. kaggle.json dosyası ASLA repo'ya commit edilmemeli (.gitignore'da).

Kullanım (Colab):

    # kaggle.json'ı Colab'a yükledikten sonra:
    python download_data.py \
        --kaggle_json /content/kaggle.json \
        --output_dir /content/statefarm

Kullanım (kaggle.json zaten ~/.kaggle/kaggle.json'da ise):

    python download_data.py --output_dir /content/statefarm
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION_SLUG = "state-farm-distracted-driver-detection"
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_JSON_TARGET = KAGGLE_CONFIG_DIR / "kaggle.json"


def setup_kaggle_credentials(kaggle_json_path: Path) -> None:
    """
    Verilen kaggle.json dosyasını Kaggle API'nin beklediği ~/.kaggle/kaggle.json
    konumuna, doğru izinlerle (600) kopyalar.
    """
    if not kaggle_json_path.exists():
        raise FileNotFoundError(
            f"kaggle.json bulunamadı: {kaggle_json_path}\n"
            f"Kaggle hesabından (Account -> API -> Create New Token) indirip "
            f"bu yola yüklediğinden emin ol."
        )

    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(kaggle_json_path, KAGGLE_JSON_TARGET)
    KAGGLE_JSON_TARGET.chmod(0o600)
    print(f"kaggle.json yerleştirildi: {KAGGLE_JSON_TARGET}")


def check_kaggle_credentials_exist() -> None:
    """
    --kaggle_json verilmediyse, ~/.kaggle/kaggle.json'ın zaten var olduğunu doğrular.
    """
    if not KAGGLE_JSON_TARGET.exists():
        raise FileNotFoundError(
            f"{KAGGLE_JSON_TARGET} bulunamadı.\n"
            f"Ya --kaggle_json parametresiyle dosya yolunu ver, "
            f"ya da kaggle.json'ı manuel olarak {KAGGLE_CONFIG_DIR} altına koy."
        )


def ensure_kaggle_cli_installed() -> None:
    """
    kaggle paketi kurulu değilse yükler. Colab'da standart pip kurulumu yeterlidir.
    Bazı "externally-managed" local Python ortamlarında (PEP 668) standart pip
    kurulumu reddedilebilir; bu durumda --break-system-packages ile tekrar denenir.
    """
    try:
        import kaggle  # noqa: F401
        return
    except ImportError:
        pass

    print("kaggle paketi bulunamadı, kuruluyor...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "externally-managed-environment" in result.stderr:
        print("Externally-managed ortam tespit edildi, --break-system-packages ile tekrar deneniyor...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet", "--break-system-packages"],
            check=True,
        )
    elif result.returncode != 0:
        raise RuntimeError(f"kaggle paketi kurulamadı:\n{result.stderr}")


def download_competition_data(output_dir: Path) -> Path:
    """
    Kaggle competition verisini indirir. Zip dosyasının yolunu döner.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{COMPETITION_SLUG}.zip"

    if zip_path.exists():
        print(f"Zip zaten mevcut, indirme atlanıyor: {zip_path}")
        return zip_path

    print(f"İndiriliyor: {COMPETITION_SLUG} -> {output_dir}")
    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION_SLUG,
            "-p", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        hint = ""
        if "403" in error_msg:
            hint = (
                "\n[İPUCU] 403 hatası genelde competition kurallarının kabul edilmediğini "
                "gösterir. https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules "
                "sayfasından 'Join Competition' yapman gerekebilir."
            )
        raise RuntimeError(f"Kaggle indirme başarısız oldu:\n{error_msg}{hint}")

    print(result.stdout)

    if not zip_path.exists():
        # Bazı durumlarda kaggle CLI farklı bir isimle indirebilir; klasördeki ilk zip'i bul
        zips = list(output_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"İndirme sonrası zip dosyası bulunamadı: {output_dir}")
        zip_path = zips[0]

    return zip_path


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Zip dosyasını açar."""
    print(f"Açılıyor: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Açma tamamlandı.")


def verify_extracted_structure(extract_to: Path) -> None:
    """
    Beklenen dosya/klasörlerin (imgs/train/c0..c9, driver_imgs_list.csv) var olup
    olmadığını kontrol eder, eksikse uyarır.
    """
    expected_csv = extract_to / "driver_imgs_list.csv"
    expected_train_dir = extract_to / "imgs" / "train"

    if not expected_csv.exists():
        print(f"[UYARI] Beklenen dosya bulunamadı: {expected_csv} — klasör yapısı farklı olabilir, kontrol et.")
    else:
        print(f"Doğrulandı: {expected_csv}")

    if not expected_train_dir.exists():
        print(f"[UYARI] Beklenen klasör bulunamadı: {expected_train_dir} — klasör yapısı farklı olabilir, kontrol et.")
    else:
        n_classes = len(list(expected_train_dir.glob("c*")))
        print(f"Doğrulandı: {expected_train_dir} ({n_classes} sınıf klasörü bulundu)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State Farm veri setini Kaggle'dan indirir.")
    parser.add_argument(
        "--kaggle_json",
        type=str,
        default=None,
        help="kaggle.json dosyasının yolu (verilmezse ~/.kaggle/kaggle.json zaten var olmalı).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Verinin indirilip açılacağı klasör (örn. /content/statefarm).",
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Belirtilirse, açma işleminden sonra zip dosyası silinmez.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    ensure_kaggle_cli_installed()

    if args.kaggle_json:
        setup_kaggle_credentials(Path(args.kaggle_json))
    else:
        check_kaggle_credentials_exist()

    zip_path = download_competition_data(output_dir)
    extract_zip(zip_path, output_dir)
    verify_extracted_structure(output_dir)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Zip silindi: {zip_path}")

    print("\n--- Tamamlandı ---")
    print(f"Veri klasörü: {output_dir}")
    print(f"driver_imgs_list.csv: {output_dir / 'driver_imgs_list.csv'}")
    print(f"Görsel klasörü: {output_dir / 'imgs' / 'train'}")


if __name__ == "__main__":
    main()
