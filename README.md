# Sürücü Dikkat/Dikkatsizlik Sınıflandırması (Driver Attention Classifier)

Araç içindeki sürücünün görsel üzerinden **attentive / distracted** olarak ikili sınıflandırılması. Model olarak Swin Transformer (Swin-Base) kullanılıyor; mimari ileride video/temporal (Video Swin Transformer) genişlemesine uyumlu olacak şekilde tasarlandı.

## Proje Yapısı

```
.
├── data/                     # Veri seti (git'e dahil değil, scriptlerle oluşturulur)
├── src/
│   └── data_prep/
│       ├── download_data.py      # Kaggle'dan State Farm veri setini indirir
│       ├── binary_mapping.py     # c0-c9 sınıflarını attentive/distracted'a indirger
│       ├── subject_split.py      # Sürücü (subject) bazlı train/val/test split
│       └── dataset.py            # PyTorch Dataset / DataLoader
├── notebooks/                # Colab notebook'ları
├── configs/                  # Eğitim konfigürasyonları
├── requirements.txt
└── .gitignore
```

## Veri Seti

**Ana eğitim seti:** [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (Kaggle)
**External validation:** AUC Distracted Driver Dataset (generalization gap ölçümü için, train'e dahil edilmiyor)

Orijinal 10 sınıf (`c0`-`c9`) ikili sınıfa indirgeniyor:
- `c0` (safe driving) → **attentive**
- `c1`-`c9` (texting, telefon, makyaj, vb.) → **distracted**

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım (Colab)

### 1. Veriyi indir

```bash
python src/data_prep/download_data.py \
    --kaggle_json /content/kaggle.json \
    --output_dir /content/statefarm
```

> Ön koşul: Kaggle hesabından API token indirilmiş olmalı (Account → API → Create New Token) ve [competition kurallarının](https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules) kabul edilmiş olması gerekir ("Join Competition").

### 2. İkili sınıfa indirge

```bash
python src/data_prep/binary_mapping.py \
    --source /content/statefarm/imgs/train \
    --target /content/statefarm_binary
```

### 3. Sürücü bazlı split oluştur

```bash
python src/data_prep/subject_split.py \
    --csv /content/statefarm/driver_imgs_list.csv \
    --output_dir /content/statefarm_splits
```

> Neden subject-based split: aynı sürücünün görüntüleri train ve test'e karışırsa model gerçekte genelleme yapmadan yapay olarak yüksek accuracy gösterebilir (data leakage).

### 4. DataLoader'ları oluştur

```python
from src.data_prep.dataset import get_dataloaders

loaders = get_dataloaders(
    splits_dir="/content/statefarm_splits",
    images_root="/content/statefarm_binary",
    batch_size=32,
    image_size=224,
)
```

## Model

- **Backbone:** Swin-Base (`swin_base_patch4_window7_224`), ImageNet pretrained (timm)
- **Neden Swin:** Hem tek-frame sınıflandırmada güçlü sonuç veriyor hem de Video Swin Transformer ailesine doğal bir geçiş sağlıyor (video fazında backbone ağırlıkları warm-start olarak yeniden kullanılabilir).

## Yol Haritası

- [x] Faz 0 — Problem tanımı
- [x] Faz 1 — Veri hazırlığı (indirme, binary mapping, subject-based split, Dataset/DataLoader)
- [ ] Faz 2 — Swin-Base fine-tuning
- [ ] Faz 3 — Değerlendirme (internal + external validation, hata analizi)
- [ ] Faz 4 — Video Swin Transformer'a genişleme

## Lisans / Veri Notu

Veri setleri repo'ya dahil edilmemiştir (Kaggle competition kuralları ve dosya boyutu limitleri nedeniyle). Yukarıdaki scriptlerle kendi ortamınızda yeniden oluşturabilirsiniz.
