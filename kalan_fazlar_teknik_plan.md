# Driver Attention Classifier — Kalan Fazlar İçin Teknik Uygulama Planı

> Bu doküman bir LLM coding agent'ın (Claude Code, Cursor, vb.) bu projeye kaldığı yerden
> devam edebilmesi için hazırlanmıştır. Her faz; hedef, girdi/çıktı, oluşturulacak dosya,
> beklenen fonksiyon imzaları, test/doğrulama kriterleri ve kabul kriterleri (acceptance
> criteria) içerecek şekilde yazılmıştır. Belirsizlik olduğunda, önceki fazlarda kurulan
> konvansiyonlar (aşağıda özetlenmiştir) takip edilmelidir.

---

## 0. Proje Bağlamı (Context)

**Problem:** Araç içi görselden ikili sınıflandırma: `attentive` (0) / `distracted` (1).
**Model ailesi:** Swin Transformer (tek-frame) → Video Swin Transformer (ileride, video fazı).
**Eğitim ortamı:** Google Colab (GPU: T4/A100).
**Repo kök dizini:** proje kökü, `src/`, `notebooks/`, `configs/`, `data/` klasörlerini içerir.

### Mevcut Durum (Faz 0-1 tamamlandı)

```
src/data_prep/
├── download_data.py     # Kaggle'dan State Farm indirir, açar, doğrular
├── binary_mapping.py    # c0-c9 -> attentive/distracted, dosya adı: {classname}_{img}
├── subject_split.py     # driver_imgs_list.csv -> subject-based train/val/test CSV
└── dataset.py           # DriverAttentionDataset, get_dataloaders()
```

**Veri akışı (zaten kurulu, değiştirilmemeli):**
```
Kaggle zip → /content/statefarm/{imgs/train/c0..c9, driver_imgs_list.csv}
           → binary_mapping.py → /content/statefarm_binary/{attentive,distracted}/{classname}_{img}
           → subject_split.py  → /content/statefarm_splits/{train,val,test}.csv
           → dataset.py get_dataloaders() → PyTorch DataLoader (train/val/test)
```

**Label encoding (sabit, değiştirilmemeli):** `LABEL_TO_IDX = {"attentive": 0, "distracted": 1}` (bkz. `dataset.py`)

**Normalizasyon:** ImageNet mean/std, `IMAGENET_MEAN = [0.485, 0.456, 0.406]`, `IMAGENET_STD = [0.229, 0.224, 0.225]` (bkz. `dataset.py`, yeni kodda da bu sabitler import edilip kullanılmalı, tekrar tanımlanmamalı).

### Kurulmuş Konvansiyonlar (yeni kodda da uygulanmalı)

1. **Dil:** Docstring ve yorumlar Türkçe, kod tanımlayıcıları (fonksiyon/değişken adı) İngilizce.
2. **CLI:** Her script `argparse` ile çalıştırılabilir olmalı (`if __name__ == "__main__":` bloğu).
3. **Test disiplini:** Her yeni script, gerçek veri olmadan **sahte/sentetik veriyle** (küçük boyutlu, üretilmiş) test edilip çalıştığı doğrulanmalı, sonra teslim edilmeli. Gerçek Kaggle verisi olmadan da script'in "syntax + logic" doğruluğu kanıtlanmalı.
4. **Hata yönetimi:** Beklenen dosya/klasör yoksa erken ve açıklayıcı hata fırlatılmalı (bkz. `dataset.py` `_validate_first_n`, `download_data.py` credential kontrolleri).
5. **Checkpoint ayrımı:** Backbone ve classification head ağırlıkları **ayrı** saklanmalı (video fazında head değişecek, backbone warm-start olarak kullanılacak).
6. **Modülerlik:** Yeni fonksiyonlar `src/` altında ilgili alt klasöre (`src/model/`, `src/train/`, `src/eval/`) eklenmeli, `dataset.py`/`binary_mapping.py` içine karıştırılmamalı.

---

## Faz 2 — Swin-Base Fine-Tuning

### 2.1 Hedef
Pretrained Swin-Base backbone'unu binary classification head ile fine-tune eden, Colab'da çalıştırılabilir bir training pipeline'ı kurmak.

### 2.2 Oluşturulacak Dosyalar

```
src/model/
├── __init__.py
└── swin_classifier.py     # Model tanımı

src/train/
├── __init__.py
└── train.py                # Training loop + CLI

configs/
└── train_config.yaml        # Hyperparametreler
```

### 2.3 `src/model/swin_classifier.py` — Beklenen İçerik

```python
class SwinDriverClassifier(nn.Module):
    def __init__(self, model_name: str = "swin_base_patch4_window7_224",
                 pretrained: bool = True, num_classes: int = 2):
        """
        timm ile pretrained Swin backbone yükler, classification head'i
        num_classes'a göre değiştirir.
        """

    def forward(self, x): ...

    def get_backbone_state_dict(self) -> dict:
        """Sadece backbone ağırlıklarını döner (head hariç) — checkpoint ayrımı için."""

    def get_head_state_dict(self) -> dict:
        """Sadece classification head ağırlıklarını döner."""
```

**Not:** `timm.create_model(model_name, pretrained=True, num_classes=num_classes)` kullanımı, backbone/head ayrımı için `model.head` (timm Swin modellerinde bu isimle geçer, doğrulanmalı) referans alınarak state_dict ayrıştırılabilir.

### 2.4 `src/train/train.py` — Beklenen İçerik

Fonksiyonlar:
- `train_one_epoch(model, loader, optimizer, criterion, device) -> dict` (loss, accuracy döner)
- `validate(model, loader, criterion, device) -> dict` (loss, accuracy, precision, recall, f1 döner — **özellikle distracted sınıfı recall'ü** raporlanmalı, bkz. proje önceliği)
- `main()` — argparse ile: `--splits_dir`, `--images_root`, `--epochs`, `--batch_size`, `--lr_backbone`, `--lr_head`, `--output_dir` (checkpoint kayıt yeri)

**Differential learning rate:** Backbone için düşük LR, head için nispeten yüksek LR (ayrı parameter group'lar ile `optimizer = AdamW([{"params": backbone_params, "lr": lr_backbone}, {"params": head_params, "lr": lr_head}])`).

**Early stopping:** Val loss veya val F1 bazlı, `patience` parametresi ile.

**Checkpoint kaydı:** Her epoch sonunda değil, en iyi val metriğinde. Ayrı dosyalar: `backbone_best.pth`, `head_best.pth`.

### 2.5 Test/Doğrulama Kriteri
- Sahte/küçük bir Dataset (örn. 20 örnek, 2 sınıf) ile 1-2 epoch'luk bir "smoke test" çalıştırılıp hata vermeden bittiği gösterilmeli.
- Loss'un epoch'lar arasında düştüğü (overfit bile olsa) doğrulanmalı — bu, forward/backward/optimizer bağlantısının doğru kurulduğunun kanıtı.

### 2.6 Kabul Kriterleri
- [ ] `SwinDriverClassifier` pretrained ağırlıkla instantiate olabiliyor
- [ ] `train_one_epoch` ve `validate` sahte veriyle hatasız çalışıyor
- [ ] Checkpoint dosyaları (backbone/head ayrı) diske yazılıyor
- [ ] CLI, Colab'da tek satırla çalıştırılabilir durumda

---

## Faz 3 — Değerlendirme ve İyileştirme

### 3.1 Hedef
Eğitilmiş modelin internal (State Farm test split) ve external (AUC dataset) performansını ölçmek, hata analizini otomatikleştirmek.

### 3.2 Oluşturulacak Dosyalar

```
src/eval/
├── __init__.py
├── evaluate.py           # Metrik hesaplama + confusion matrix
├── error_analysis.py      # Yanlış sınıflandırılan örnekleri kategorize etme
└── gradcam_viz.py         # Grad-CAM / attention rollout görselleştirme
```

### 3.3 `evaluate.py` — Beklenen İçerik

- `run_evaluation(model, loader, device) -> dict`: accuracy, precision, recall, f1, confusion_matrix (sklearn ile) döner.
- `evaluate_external(model, external_splits_dir, external_images_root, device) -> dict`: AUC dataset üzerinde aynı metrikleri hesaplar. **Internal vs external metrik farkını** (generalization gap) ayrı bir rapor olarak yazdırır.
- CLI: `--checkpoint_backbone`, `--checkpoint_head`, `--splits_dir`, `--images_root`, `--external_splits_dir` (opsiyonel), `--output_json` (sonuçları JSON'a yazma).

### 3.4 `error_analysis.py` — Beklenen İçerik

- `binary_mapping.py`'nin dosya adına gömdüğü orijinal sınıf bilgisini (`{classname}_{img}` prefix'i) kullanarak, yanlış sınıflandırılan `distracted` örneklerinin **hangi orijinal alt-davranıştan** (c1-c9) geldiğini say, tablo olarak raporla.
- Çıktı: `error_breakdown.csv` (kolonlar: `original_class, true_label, predicted_label, count`)

### 3.5 `gradcam_viz.py` — Beklenen İçerik

- Swin Transformer için attention rollout veya Grad-CAM benzeri bir görselleştirme (timm modelleri için `pytorch-grad-cam` kütüphanesi kullanılabilir, Swin'in windowed attention yapısına uygun target layer seçilmeli).
- Yanlış sınıflandırılan birkaç örnek için görsel + heatmap kaydet (`output_dir/gradcam_samples/`).

### 3.6 Kabul Kriterleri
- [ ] `run_evaluation` sahte veriyle çalışıp beklenen anahtarları (accuracy, precision, recall, f1, confusion_matrix) içeren dict döndürüyor
- [ ] External evaluation, internal ile aynı fonksiyon imzasını paylaşıyor (kod tekrarı yok)
- [ ] `error_breakdown.csv` doğru şekilde orijinal sınıf bilgisini geri çıkarabiliyor
- [ ] Grad-CAM çıktısı görsel olarak kaydediliyor (en az 1 örnek ile test edilmiş)

---

## Faz 4 — Video Swin Transformer'a Genişleme

### 4.1 Hedef
Faz 2'de eğitilen Swin-Base backbone'unu warm-start olarak kullanarak, klip-tabanlı (video) sınıflandırmaya geçiş.

### 4.2 Ön Koşul (bu fazdan önce netleştirilmeli)
- Video veri seti seçimi: AI City Challenge 2024 Track 3 veya DMD (Driver Monitoring Dataset)
- Klip uzunluğu kararı (öneri: 16 kare, CogDrive konvansiyonuyla tutarlı)

### 4.3 Oluşturulacak Dosyalar

```
src/data_prep/
└── video_clip_dataset.py   # Klip-tabanlı Dataset (video_id + frame sekansı gruplama)

src/model/
└── video_swin_classifier.py  # Video Swin Transformer, backbone warm-start ile
```

### 4.4 `video_clip_dataset.py` — Beklenen İçerik
- Faz 1'de bahsedilen "video_id, frame_no" organizasyon notunu temel alarak, ham videoyu/karelerini `num_frames`'lik kliplere böler.
- `VideoClipDataset(Dataset)`: `__getitem__` bir klip (T, C, H, W) tensor + label döner.

### 4.5 `video_swin_classifier.py` — Beklenen İçerik
- `timm` veya `torchvision.models.video` üzerinden Video Swin Transformer (ya da benzer bir 3D windowed attention modeli) yükler.
- **Kritik:** Faz 2'de kaydedilen `backbone_best.pth` ağırlıklarının, video modelinin 2D spatial katmanlarına nasıl aktarılacağı (weight inflation / spatial-to-spatiotemporal mapping) fonksiyon olarak yazılmalı: `load_pretrained_2d_backbone(video_model, path_to_2d_checkpoint)`.

### 4.6 Kabul Kriterleri
- [ ] Klip-tabanlı Dataset, sahte video (birkaç sahte kare dizisi) ile test edilmiş
- [ ] 2D backbone ağırlıklarının video modeline aktarımı hatasız çalışıyor (en azından şekil uyumsuzluğu hatası vermiyor)
- [ ] Küçük bir smoke-test eğitimi (1 epoch, sahte klip verisiyle) hatasız tamamlanıyor

---

## Genel Notlar (Tüm Fazlar İçin)

- **Colab uyumluluğu:** Her script'in Colab'da GPU olsun/olmasın (`device = "cuda" if torch.cuda.is_available() else "cpu"`) çalışabilmesi gerekir; sadece test/smoke-test aşamasında CPU'da da çalışmalı.
- **Reproducibility:** Random seed (`torch.manual_seed`, `numpy.random.seed`) her training/eval script'inde sabitlenmeli.
- **Tez bağlamı:** Bu proje TÜBİTAK 2209-A kapsamında değil, ayrı bir kişisel/portfolyo projesi olarak yürütülüyor — ancak segmentasyon tez projesindeki metodolojik disiplin (subject-based split, internal/external validation ayrımı, negatif sonuçların şeffaf raporlanması) aynı şekilde uygulanmalı.
- **Belirsizlik durumunda:** Yeni bir mimari/kütüphane kararı (ör. hangi Video Swin implementasyonu) alınmadan önce, mevcut `README.md`'deki yol haritası ve bu dokümandaki kabul kriterleri referans alınmalı; kapsam dışına çıkan büyük mimari değişiklikler için kullanıcıya danışılmalı.
