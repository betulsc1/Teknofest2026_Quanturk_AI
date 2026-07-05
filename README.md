# TEKNOFEST 2026 — Havacılıkta Yapay Zekâ Yarışması

> **Takım:** QUANTÜRK  
> **Yarışma:** TEKNOFEST 2026 Havacılıkta Yapay Zekâ  
> **Şartname Versiyonu:** V1.0 (21.02.2026)

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Görevler](#görevler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Model Eğitimi](#model-eğitimi)
- [Yarışma Modu](#yarışma-modu)
- [Test ve Değerlendirme](#test-ve-değerlendirme)
- [Katkı Sağlayanlar](#katkı-sağlayanlar)

---

## Proje Hakkında

Bu proje, TEKNOFEST 2026 Havacılıkta Yapay Zekâ Yarışması için geliştirilmiş bir görüntü işleme ve yapay zekâ sistemidir. Drone'un alt-görüş kamerasından alınan görüntüler üzerinde gerçek zamanlı nesne tespiti, pozisyon kestirimi ve görüntü eşleme yapılmaktadır.

### Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| İşletim Sistemi | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3060 (8 GB VRAM) | NVIDIA RTX 4090 (24 GB VRAM) |
| RAM | 16 GB | 32 GB |
| Depolama | 256 GB SSD | 1 TB NVMe SSD |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.1 |

---

## Görevler

### Görev 1 — Nesne Tespiti (%25)
Drone görüntülerinde **4 nesne sınıfı** tespit edilmektedir:

| Sınıf ID | Sınıf | Ek Bilgi |
|----------|-------|----------|
| 0 | Taşıt | Hareket durumu (0=hareketsiz, 1=hareketli) |
| 1 | İnsan | — |
| 2 | UAP (Uçan Araba Park Alanı) | İniş durumu (0=uygun değil, 1=uygun) |
| 3 | UAİ (Uçan Ambulans İniş Alanı) | İniş durumu (0=uygun değil, 1=uygun) |

**Kullanılan Teknolojiler:**
- YOLOv9-C — Ana tespit modeli
- RT-DETR-L — Ensemble için ikincil model
- ByteTracker — Nesne takibi ve hareket analizi
- Homografi Kompanzasyonu — Kamera hareketi ile gerçek nesne hareketini ayırt etme

### Görev 2 — Pozisyon Tespiti (%40)
GPS sisteminin devre dışı kaldığı senaryolarda drone'un X/Y/Z pozisyonunu yalnızca kamera görüntüleri ile kestirme.

**Kullanılan Teknolojiler:**
- RAFT (Recurrent All-Pairs Field Transforms) — Optik akış
- Extended Kalman Filter (EKF) — Pozisyon füzyonu ve gürültü azaltma
- Pinhole Camera Modeli — Piksel → Metre dönüşümü

### Görev 3 — Görüntü Eşleme (%25)
Oturum başında verilen referans nesnelerin drone görüntülerinde anlık olarak tespiti.

**Kullanılan Teknolojiler:**
- SuperPoint — Keypoint çıkarma
- SuperGlue — Feature eşleştirme
- DINOv2 ViT-L/14 — Cross-modal (termal↔RGB) semantic eşleştirme
- RANSAC — Geometrik doğrulama

---

## Kurulum

### 1. Repoyu Klonla

```bash
git clone https://github.com/[takim-adi]/teknofest2026-ai.git
cd teknofest2026-ai
```

### 2. Conda Ortamı Oluştur (Önerilen)

```bash
conda create -n teknofest python=3.11 -y
conda activate teknofest
```

### 3. CUDA ve PyTorch Kur

```bash
# CUDA 12.1 için:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 için:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 5. RAFT Model Ağırlıklarını İndir

```bash
python scripts/setup_environment.py --download-models
```

### 6. SuperPoint & SuperGlue Ağırlıklarını İndir

```bash
# SuperGlue resmi reposunu kur
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
cp -r SuperGluePretrainedNetwork/models/weights/ models/task3/
```

### 7. Kurulum Doğrulaması

```bash
python scripts/setup_environment.py --verify
```

Başarılı çıktı şöyle görünmeli:
```
[OK] Python 3.11
[OK] CUDA 12.1 — GPU: NVIDIA RTX XXXX
[OK] PyTorch 2.1.0
[OK] YOLOv9 model yüklendi
[OK] RAFT model yüklendi
[OK] SuperGlue model yüklendi
[OK] DINOv2 model yüklendi
Kurulum başarılı. Hazır.
```

---

## Konfigürasyon

Yarışma öncesinde `config/` klasöründeki dosyalar düzenlenmelidir.

### config/camera_params.yaml

```yaml
# Kamera intrinsic parametreleri (yarışma günü paylaşılacak)
camera:
  fx: 1000.0        # X ekseni odak uzaklığı (piksel)
  fy: 1000.0        # Y ekseni odak uzaklığı (piksel)
  cx: 960.0         # Optik merkez X (genellikle genişlik/2)
  cy: 540.0         # Optik merkez Y (genellikle yükseklik/2)
  width: 1920       # Görüntü genişliği (piksel)
  height: 1080      # Görüntü yüksekliği (piksel)
  distortion:       # Lens distorsiyon katsayıları
    k1: 0.0
    k2: 0.0
    p1: 0.0
    p2: 0.0
```

> ⚠️ Bu değerler yarışmada organizatörler tarafından paylaşılacaktır. Paylaşılana kadar varsayılan değerler kullanılır.

### config/server_config.yaml

```yaml
# Yarışma günü doldurulacak
server:
  url: "http://127.0.0.25:5000"   # Test için örnek
  token: "YOUR_TEAM_TOKEN_HERE"
  timeout: 5                       # saniye
  max_retries: 3
```

---

## Kullanım

### Test Oturumu (Yarışma Öncesi 75 Dakikalık Test)

```bash
python scripts/run_test_session.py \
    --server-url http://[sunucu-ip]:5000 \
    --token [takim-token]
```

### Yarışma Modu

```bash
python scripts/run_competition.py \
    --server-url http://[sunucu-ip]:5000 \
    --token [takim-token] \
    --session-id [oturum-id]
```

### Sadece Tek Görevi Test Et

```bash
# Görev 1 test
python scripts/run_competition.py --task 1 --debug

# Görev 2 test
python scripts/run_competition.py --task 2 --debug

# Görev 3 test (referans görsel klasörü gerekli)
python scripts/run_competition.py --task 3 \
    --references data/reference_objects/ --debug
```

### Debug Görselleştirme

```bash
# Gerçek zamanlı bbox çizimi ile çalıştır
python scripts/run_competition.py --visualize

# Kayıt alarak çalıştır
python scripts/run_competition.py --visualize --save-video output.mp4
```

---

## Proje Yapısı

```
teknofest2026_ai/
│
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
│
├── config/
│   ├── config.yaml              # Genel ayarlar ve hiperparametreler
│   ├── camera_params.yaml       # Kamera kalibrasyonu
│   └── server_config.yaml       # Sunucu bağlantı ayarları
│
├── data/                        # Veri (git'e eklenmez — .gitignore)
│   ├── raw/
│   ├── processed/
│   ├── augmented/
│   ├── datasets/
│   │   ├── task1/
│   │   ├── task2/
│   │   └── task3/
│   └── reference_objects/
│
├── models/                      # Model ağırlıkları (git'e eklenmez)
│   ├── task1/
│   ├── task2/
│   └── task3/
│
├── src/
│   ├── core/                    # Ana pipeline bileşenleri
│   ├── task1_detection/         # Nesne tespiti modülü
│   ├── task2_position/          # Pozisyon kestirimi modülü
│   ├── task3_matching/          # Görüntü eşleme modülü
│   ├── communication/           # Sunucu iletişim katmanı
│   └── utils/                   # Yardımcı araçlar
│
├── training/                    # Eğitim scriptleri
│   ├── data_preparation/
│   └── evaluation/
│
├── scripts/                     # Çalıştırma scriptleri
└── tests/                       # Unit testler
```

---

## Model Eğitimi

### Görev 1 — Nesne Tespiti Eğitimi

```bash
# Önce veri setlerini hazırla
python training/data_preparation/download_datasets.py
python training/data_preparation/convert_to_yolo.py
python training/data_preparation/split_dataset.py

# Eğitimi başlat
python training/train_task1.py \
    --model yolov9c \
    --data data/datasets/task1/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 1280 \
    --device 0
```

### Görev 2 — Pozisyon Tespiti (Eğitim gerekmez)

RAFT modeli önceden eğitilmiş ağırlıklar kullanır. Sadece kamera parametrelerinin doğru ayarlanması yeterlidir.

```bash
# Kamera parametrelerini doğrula
python training/train_task2.py --calibrate \
    --camera-params config/camera_params.yaml
```

### Görev 3 — Görüntü Eşleme (Eğitim gerekmez)

SuperGlue ve DINOv2 önceden eğitilmiş modeller kullanır. Fine-tuning opsiyoneldir.

---

## Test ve Değerlendirme

### Görev 1 — mAP Hesaplama

```bash
python training/evaluation/eval_task1.py \
    --weights models/task1/detector/best.pt \
    --data data/datasets/task1/dataset.yaml \
    --iou-threshold 0.5
```

Çıktı örneği:
```
Sınıf          AP@0.5   AP@0.5:0.95
Taşıt          0.812    0.543
İnsan          0.743    0.478
UAP            0.891    0.712
UAİ            0.876    0.698
mAP@0.5        0.831
mAP@0.5:0.95   0.608
```

### Görev 2 — Pozisyon Hata Metriği

```bash
python training/evaluation/eval_task2.py \
    --video data/test_videos/sample.mp4 \
    --ground-truth data/test_videos/sample_gt.csv
```

Çıktı örneği:
```
Ortalama Hata (E): 3.24 metre
Maksimum Hata:     12.7 metre
GPS Sağlıklı Süre: 60 sn → Hata: 0.8 m
GPS Sağlıksız Süre: 240 sn → Hata: 4.1 m
```

### Benchmark (Hız Testi)

```bash
python scripts/benchmark.py --frames 100
```

```
Pipeline Hız Testi (100 frame):
  Görev 1 (Tespit): 18.3 ms/frame
  Görev 2 (Pozisyon): 31.2 ms/frame
  Görev 3 (Eşleme): 12.7 ms/frame
  JSON Hazırlama:    2.1 ms/frame
  Toplam:           64.3 ms/frame → 15.6 FPS ✓
  Hedef (7.5 FPS):  133 ms/frame  → BAŞARILI
```

---

## Kullanılan Veri Setleri

| Veri Seti | Amaç | Kaynak |
|-----------|------|--------|
| VisDrone2019 | Taşıt ve insan tespiti (drone) | github.com/VisDrone |
| DOTA v2 | Kuş bakışı nesne tespiti | captain-whu.github.io/DOTA |
| VEDAI | Araç tespiti (kuş bakışı) | downloads.greyc.fr/vedai |
| AU-AIR | Drone uçuş verisi | bozcuoglu.com/au-air |

---

## Katkı Sağlayanlar

| İsim | Rol |
|------|-----|
| [İsim 1] | Takım Kaptanı — Sistem Mimarisi |
| [İsim 2] | Görev 1 — Nesne Tespiti |
| [İsim 3] | Görev 2 — Pozisyon Kestirimi |
| [İsim 4] | Görev 3 — Görüntü Eşleme |
| [İsim 5] | Sunucu İletişim & Entegrasyon |

---

## Lisans

Bu proje yalnızca TEKNOFEST 2026 yarışması kapsamında kullanılmak üzere geliştirilmiştir.

---

*Son güncelleme: Mart 2026*
