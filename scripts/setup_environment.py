"""
scripts/setup_environment.py

Yarışma öncesi çalıştırılır:
  python scripts/setup_environment.py --verify
  python scripts/setup_environment.py --download-models
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"  {'[OK]' if ok else '[!!]'} Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_cuda():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA {torch.version.cuda} — GPU: {name}")
        else:
            print("  [!!] CUDA bulunamadı — CPU kullanılacak (yavaş!)")
        return cuda_ok
    except ImportError:
        print("  [!!] PyTorch kurulu değil")
        return False


def check_models():
    paths = [
        ROOT / "models/task1/detector/best.pt",
        ROOT / "models/task3/superglue/superglue_outdoor.pth",
    ]
    all_ok = True
    for p in paths:
        ok = p.exists()
        print(f"  {'[OK]' if ok else '[!!]'} {p.relative_to(ROOT)}")
        all_ok = all_ok and ok
    return all_ok


def check_imports():
    libs = [
        ("ultralytics",  "YOLOv9"),
        ("cv2",          "OpenCV"),
        ("filterpy",     "EKF"),
        ("albumentations","Augmentation"),
        ("requests",     "HTTP"),
        ("yaml",         "YAML"),
    ]
    all_ok = True
    for lib, label in libs:
        try:
            __import__(lib)
            print(f"  [OK] {label} ({lib})")
        except ImportError:
            print(f"  [!!] {label} ({lib}) — pip install {lib}")
            all_ok = False
    return all_ok


def download_models():
    """RAFT ve diğer pretrained ağırlıkları indir."""
    print("\nModel ağırlıkları indiriliyor...")

    # RAFT — torchvision ile otomatik gelir, elle indirmeye gerek yok
    print("  RAFT: torchvision ile otomatik yüklenir ✓")

    # DINOv2
    print("  DINOv2: torch.hub ile ilk çalıştırmada otomatik indirilir ✓")

    # SuperGlue — repodan manuel kopyalanması gerekir
    sg_path = ROOT / "models/task3/superglue"
    sg_path.mkdir(parents=True, exist_ok=True)
    if not (sg_path / "superglue_outdoor.pth").exists():
        print("  [!!] SuperGlue ağırlıkları eksik.")
        print("       Adımlar:")
        print("       1) git clone https://github.com/magicleap/SuperGluePretrainedNetwork")
        print("       2) cp SuperGluePretrainedNetwork/models/weights/*.pth models/task3/superglue/")
    else:
        print("  [OK] SuperGlue ağırlıkları mevcut")

    # Klasör yapısını oluştur
    for d in ["data/raw", "data/processed", "data/reference_objects",
              "models/task1/detector", "models/task2/flow_estimator"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("  [OK] Klasör yapısı oluşturuldu")


def verify():
    print("\n=== TEKNOFEST 2026 — Ortam Doğrulama ===\n")

    results = {
        "Python":    check_python(),
        "CUDA/GPU":  check_cuda(),
        "Kütüphaneler": check_imports(),
        "Modeller":  check_models(),
    }

    print("\n--- Özet ---")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'✓' if ok else '✗ HATA'}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ Sistem hazır. Yarışmaya başlayabilirsin!\n")
    else:
        print("\n✗ Bazı sorunlar var. Yukarıdaki [!!] işaretlerini çöz.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--verify",          action="store_true")
    p.add_argument("--download-models", action="store_true")
    args = p.parse_args()

    if args.download_models:
        download_models()
    if args.verify or not any(vars(args).values()):
        verify()