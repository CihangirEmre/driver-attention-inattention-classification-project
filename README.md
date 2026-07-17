# Sürücü Dikkat/Dikkatsizlik Sınıflandırması (Driver Attention Classifier)

Araç içindeki sürücünün görsel üzerinden **attentive / distracted** olarak ikili sınıflandırılması. Model olarak Swin Transformer (Swin-Base) kullanılıyor; mimari ileride video/temporal (Video Swin Transformer) genişlemesine uyumlu olacak şekilde tasarlandı.

## Proje Yapısı

```
.
├── data/                               # Veri seti (git'e dahil değil, scriptlerle oluşturulur)
├── src/
│   ├── data_prep/
│   │   ├── download_data.py            # Kaggle'dan State Farm veri setini indirir
│   │   ├── binary_mapping.py           # c0-c9 sınıflarını attentive/distracted'a indirger
│   │   ├── subject_split.py            # Sürücü (subject) bazlı train/val/test split
│   │   ├── dataset.py                  # PyTorch Dataset / DataLoader
│   │   └── prepare_external_dataset.py # External veri setini (Roboflow vb.) ikili formata çevirir
│   ├── model/
│   │   └── swin_classifier.py          # Swin-Base tabanlı binary classifier (backbone/head ayrımı)
│   ├── train/
│   │   └── train.py                    # Fine-tuning training loop + CLI
│   └── eval/
│       └── evaluate.py                 # Internal/external değerlendirme + generalization gap
├── configs/
│   └── train_config.yaml               # Eğitim hyperparametreleri
├── notebooks/                          # Colab notebook'ları
├── requirements.txt
└── .gitignore
```

## Veri Seti

**Ana eğitim seti:** [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (Kaggle)

**External validation:** [Distracted Driver Dataset](https://universe.roboflow.com/testing-g0qqv/distracted-driver-py6vx/dataset/1) (Roboflow Universe). Planlanan AUC Distracted Driver Dataset yerine bu kullanıldı çünkü AUC erişimi manuel bir lisans/onay süreci gerektiriyor; Roboflow üzerindeki bu veri seti doğrudan indirilebiliyor ve State Farm ile aynı `c0`-`c9` sınıf isimlendirmesini kullanıyor (tüm sınıflar mevcut olmayabilir, script eksik sınıfları otomatik atlar).

Orijinal sınıflar (`c0`-`c9`) ikili sınıfa indirgeniyor:
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

### 4. Modeli eğit

```bash
python src/train/train.py --config configs/train_config.yaml
```

- **Differential learning rate:** backbone için düşük (`lr_backbone`), head için yüksek (`lr_head`) LR, ayrı optimizer parameter group'larıyla.
- **Early stopping:** val F1 (distracted sınıfı) bazlı, `patience` epoch boyunca iyileşme olmazsa durur.
- **Checkpoint kaydı:** sadece en iyi val F1'de, backbone ve head ayrı dosyalara (`backbone_best.pth`, `head_best.pth`) — Faz 4'te (video) backbone'un warm-start olarak yeniden kullanılabilmesi için.

CLI argümanları `--config` dosyasındaki değerleri override edebilir (örn. `--epochs 5`).

### 5. Modeli değerlendir (internal test set)

```bash
python src/eval/evaluate.py \
    --checkpoint_backbone /content/checkpoints/backbone_best.pth \
    --checkpoint_head /content/checkpoints/head_best.pth \
    --splits_dir /content/statefarm_splits \
    --images_root /content/statefarm_binary \
    --output_json /content/eval_results.json
```

### 6. External veri setiyle generalization gap ölç (opsiyonel)

```bash
python src/data_prep/prepare_external_dataset.py \
    --source /content/external_raw \
    --images_root /content/external_binary \
    --splits_dir /content/external_splits

python src/eval/evaluate.py \
    --checkpoint_backbone /content/checkpoints/backbone_best.pth \
    --checkpoint_head /content/checkpoints/head_best.pth \
    --external_splits_dir /content/external_splits \
    --external_images_root /content/external_binary
```

Internal ve external argümanları (`--splits_dir`/`--images_root` ve `--external_splits_dir`/`--external_images_root`) birbirinden bağımsızdır — sadece external test istiyorsan internal argümanları hiç vermeyebilirsin. İkisi de verilirse aralarındaki generalization gap ayrıca raporlanır.

## Model

- **Backbone:** Swin-Base (`swin_base_patch4_window7_224`), ImageNet pretrained (timm)
- **Neden Swin:** Hem tek-frame sınıflandırmada güçlü sonuç veriyor hem de Video Swin Transformer ailesine doğal bir geçiş sağlıyor (video fazında backbone ağırlıkları warm-start olarak yeniden kullanılabilir).

## Sonuçlar

Swin-Base, State Farm train/val split'i üzerinde fine-tune edilip (early stopping ile en iyi val F1'de durduruldu) hem State Farm test split'inde (internal) hem de bağımsız bir Roboflow veri setinde (external) değerlendirildi:

| Metrik | Internal (State Farm test, n=3525) | External (Roboflow, n=2000) |
|---|---|---|
| Accuracy | 94.24% | 81.50% |
| Precision (distracted) | 96.46% | 96.73% |
| Recall (distracted) | 97.05% | 82.22% |
| F1 (distracted) | 96.75% | 88.89% |

**Generalization gap:** External veri setinde recall ~14.8 puan düşüyor (97.05% → 82.22%) — model, farklı kamera/ortam koşullarındaki distracted örneklerin belirgin bir kısmını kaçırıyor. Precision'ın external'de yüksek kalması (96.73%), modelin "distracted" dediğinde genelde haklı olduğunu ama bazı gerçek distracted durumları "attentive" sanarak kaçırdığını gösteriyor. Bu, projenin öncelik verdiği metrik (distracted recall) açısından şeffafça raporlanması gereken bir sınırlama.

## Yol Haritası

- [x] Faz 0 — Problem tanımı
- [x] Faz 1 — Veri hazırlığı (indirme, binary mapping, subject-based split, Dataset/DataLoader)
- [x] Faz 2 — Swin-Base fine-tuning
- [ ] Faz 3 — Değerlendirme (`evaluate.py` ile internal + external + generalization gap tamamlandı; hata analizi (`error_analysis.py`) ve Grad-CAM görselleştirme (`gradcam_viz.py`) henüz yapılmadı)
- [ ] Faz 4 — Video Swin Transformer'a genişleme

## Lisans / Veri Notu

Veri setleri repo'ya dahil edilmemiştir (Kaggle competition kuralları, external dataset lisansları ve dosya boyutu limitleri nedeniyle). Yukarıdaki scriptlerle kendi ortamınızda yeniden oluşturabilirsiniz.
