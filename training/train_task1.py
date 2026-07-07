"""
Görev 1 — YOLOv8/v9 Fine-Tuning Scripti (2026 Verileri İçin)
4K drone görüntülerinde taşıt, insan, UAP, UAİ tespiti

Kullanım:
    # Standart fine-tuning (yeni model 2026_best.pt olarak kaydedilir):
    python training/train_task1.py --weights models/task1/detector/2025_best.pt --data data/datasets/task1/dataset_2026.yaml

    # Eğitim + doğrulama sonrası modeli çıkarım için aktif et (best.pt üzerine yaz):
    python training/train_task1.py --promote

Model dosya düzeni (models/task1/detector/):
    2025_best.pt  -> fine-tuning için baz alınan eski model
    2026_best.pt  -> bu script ile üretilen yeni fine-tuned model
    best.pt       -> çıkarımda (run_competition.py) kullanılan AKTİF model
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='yolov8m',
                   choices=['yolov8m', 'yolov8l', 'yolov8x', 'yolov9c', 'yolov9e'])
    # 2025'te elde edilen en iyi model ağırlıkları (fine-tuning baz modeli)
    p.add_argument('--weights', default='models/task1/detector/2025_best.pt',
                   help='Fine-tuning için baz alınacak eski model ağırlıkları')
    p.add_argument('--data',    default='data/datasets/task1/dataset_2026.yaml')
    p.add_argument('--promote', action='store_true',
                   help='Eğitim bitince yeni modeli çıkarımda kullanılan best.pt üzerine yazar')
    p.add_argument('--epochs',  type=int, default=30)  # Fine-tuning için düşürüldü
    p.add_argument('--batch',   type=int, default=4)   # T4 16GB + 4K → 4
    p.add_argument('--imgsz',   type=int, default=1280)
    p.add_argument('--device',  default='0')
    p.add_argument('--resume',  default=None,
                   help='Yarım kalan eğitimi devam ettir: runs/.../last.pt')
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        print(f"🔄 Eğitim devam ettiriliyor: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    print(f"🚀 Fine-Tuning Başlıyor: {args.model}")
    print(f"   Baz Model (2025): {args.weights}")
    print(f"   Yeni Veri (2026): {args.data}")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")

    # DEĞİŞTİRİLDİ: Boş model yerine Quantürk'ün önceden eğittiği model yükleniyor
    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        project='runs/task1',
        name=f'{args.model}_teknofest_2026_finetune',
        
        # --- FINE-TUNING İÇİN KRİTİK AYARLAR ---
        freeze=10,         # İlk 10 katmanı dondurarak eski bilgiyi koru
        lr0=0.0001,        # 0.001'den daha düşük bir öğrenme oranı (hassas ayar)
        lrf=0.01,
        cos_lr=True,       # Cosine Annealing ile yumuşak geçiş
        patience=10,       # Overfitting'i erken yakalamak için sınır düşürüldü

        # --- Drone görüntüsüne özel augmentation (Aynı Bırakıldı) ---
        flipud=0.3,      
        fliplr=0.5,
        degrees=15,      
        translate=0.1,
        scale=0.5,       
        mosaic=1.0,      
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,       

        # --- Optimizer ---
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3,

        # --- Kayıt ---
        save=True,
        save_period=5,   # Epoch azaldığı için daha sık checkpoint alınır
        plots=True,
        val=True,
    )

    # En iyi modeli kanonik models/task1/detector/ klasörüne kopyala
    best = Path(f'runs/task1/{args.model}_teknofest_2026_finetune/weights/best.pt')
    if best.exists():
        import shutil
        model_dir = Path('models/task1/detector')
        model_dir.mkdir(parents=True, exist_ok=True)

        dst = model_dir / '2026_best.pt'
        shutil.copy2(best, dst)
        print(f"\n✅ 2026 verileriyle güncellenmiş yeni en iyi model kaydedildi: {dst}")

        if args.promote:
            active = model_dir / 'best.pt'
            shutil.copy2(best, active)
            print(f"🚀 Model çıkarım için aktif edildi (run_competition.py bunu kullanır): {active}")
        else:
            print("ℹ️  Modeli çıkarımda kullanmak için doğrulamadan sonra --promote ile tekrar çalıştır")
            print("   veya elle kopyala: models/task1/detector/2026_best.pt -> models/task1/detector/best.pt")
    else:
        print(f"\n⚠️  Eğitim çıktısı bulunamadı: {best}")


if __name__ == '__main__':
    main()
