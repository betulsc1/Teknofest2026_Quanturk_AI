#!/usr/bin/env python3
"""
data/datasets/task2/generate_synthetic_dataset.py

Teknofest Görev 2 yarışma ortamını birebir simüle eden sentetik veri seti üreteci.

Üretilen dosyalar:
    frames/          → 1920x1080 PNG kareler (drone alt kamerasından görüntü)
    ground_truth.json → Her kare için gerçek (x,y,z) ve gps_health_status
    metadata.json    → Veri seti meta bilgileri

Senaryo:
    - 3 dakika uçuş, 7.5 FPS → 1350 kare
    - İlk 1 dakika (450 kare): GPS sağlıklı (health_status=1)
    - Sonraki 2 dakika (900 kare): GPS kesik (health_status=0)
    - Drone düz değil, "8" çizen bir yörünge izler (gerçekçi test)
    - Zemin dokusu: Perlin-benzeri gürültü + çizgiler + geometrik şekiller
      (Optik akışın tutunabileceği zengin doku)

Kullanım:
    python generate_synthetic_dataset.py
"""

import json
import os
import sys
import math
import time
import numpy as np
import cv2

# ─── Konfigürasyon ───────────────────────────────────────────────────── #

FPS = 7.5
DURATION_SEC = 300          # 5 dakika (Şartname: toplam 5 dakika oturum)
TOTAL_FRAMES = int(FPS * DURATION_SEC)  # 2250
GPS_HEALTHY_SEC = 60        # İlk 60 saniye sağlıklı (Şartname: ilk 1 dk kesin sağlıklı)
GPS_HEALTHY_FRAMES = int(FPS * GPS_HEALTHY_SEC)  # 450

# Kamera parametreleri (yarışma benzeri)
FRAME_W, FRAME_H = 1920, 1080
FX, FY = 1000.0, 1000.0
CX, CY = FRAME_W / 2.0, FRAME_H / 2.0

# Zemin haritası boyutları (drone'un tüm yörüngesini kapsayacak)
MAP_W, MAP_H = 10000, 10000

# Drone irtifası (metre)
ALTITUDE = 50.0

# Uçuş yörünge parametreleri
# Drone bir "Lissajous" (8 şekli) çizer
ORBIT_A = 40.0     # X ekseni amplitüd (metre)
ORBIT_B = 25.0     # Y ekseni amplitüd (metre)
ORBIT_PERIOD = 120.0  # Tam bir 8 çizme süresi (saniye)

# ─── Mock Landmark (UAP) ─────────────────────────────────────────────── #
NUM_LANDMARKS = 15
LANDMARK_SIZE_M = 4.5  # UAP çapı (metre)
# Randomly generated fixed landmarks will be stored here
LANDMARKS = []

# Drone irtifası (metre)
ALTITUDE = 50.0

# Uçuş yörünge parametreleri
# Drone bir "Lissajous" (8 şekli) çizer
ORBIT_A = 40.0     # X ekseni amplitüd (metre)
ORBIT_B = 25.0     # Y ekseni amplitüd (metre)
ORBIT_PERIOD = 120.0  # Tam bir 8 çizme süresi (saniye)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Zemin Haritası Üretimi ──────────────────────────────────────────── #

def generate_ground_map(w: int, h: int) -> np.ndarray:
    """
    Optik akış için zengin dokulu zemin haritası üretir.
    Gerçek dünya zemini gibi: yollar, binalar, ağaç desenleri.
    """
    print("[1/4] Zemin haritası üretiliyor...")
    ground = np.zeros((h, w, 3), dtype=np.uint8)

    # Katman 1: Rastgele gürültü arka plan (çimen benzeri)
    np.random.seed(42)
    noise = np.random.randint(30, 80, (h, w, 3), dtype=np.uint8)
    ground = noise.copy()

    # Katman 2: Büyük rastgele renkli yamalar (binalar/alanlar)
    for _ in range(200):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        bw = np.random.randint(50, 300)
        bh = np.random.randint(50, 300)
        color = (
            np.random.randint(40, 200),
            np.random.randint(40, 200),
            np.random.randint(40, 200),
        )
        cv2.rectangle(ground, (x1, y1), (x1 + bw, y1 + bh), color, -1)

    # Katman 3: Yollar (yatay ve dikey çizgiler)
    for _ in range(15):
        if np.random.random() > 0.5:
            # Yatay yol
            y = np.random.randint(100, h - 100)
            thickness = np.random.randint(8, 25)
            cv2.line(ground, (0, y), (w, y), (90, 90, 90), thickness)
            # Yol çizgisi
            cv2.line(ground, (0, y), (w, y), (200, 200, 200), 2)
        else:
            # Dikey yol
            x = np.random.randint(100, w - 100)
            thickness = np.random.randint(8, 25)
            cv2.line(ground, (x, 0), (x, h), (90, 90, 90), thickness)
            cv2.line(ground, (x, 0), (x, h), (200, 200, 200), 2)

    # Katman 4: Küçük nesneler (araçlar, ağaçlar benzeri)
    for _ in range(500):
        cx, cy = np.random.randint(50, w - 50), np.random.randint(50, h - 50)
        shape_type = np.random.choice(["circle", "rect", "triangle"])
        color = (
            np.random.randint(20, 255),
            np.random.randint(20, 255),
            np.random.randint(20, 255),
        )
        if shape_type == "circle":
            r = np.random.randint(5, 30)
            cv2.circle(ground, (cx, cy), r, color, -1)
        elif shape_type == "rect":
            rw, rh = np.random.randint(5, 40), np.random.randint(5, 40)
            cv2.rectangle(ground, (cx, cy), (cx + rw, cy + rh), color, -1)
        else:
            pts = np.array([
                [cx, cy - np.random.randint(5, 20)],
                [cx - np.random.randint(5, 20), cy + np.random.randint(5, 20)],
                [cx + np.random.randint(5, 20), cy + np.random.randint(5, 20)],
            ])
            cv2.fillPoly(ground, [pts], color)

    # Katman 5: Benzersiz işaretçiler (LightGlue loop closure için)
    for i in range(30):
        mx = np.random.randint(200, w - 200)
        my = np.random.randint(200, h - 200)
        cv2.putText(
            ground, f"M{i:02d}",
            (mx, my), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (255, 255, 0), 4,
        )
        cv2.circle(ground, (mx, my), 40, (0, 255, 255), 3)

    return ground


# ─── Uçuş Yörüngesi ─────────────────────────────────────────────────── #

def compute_trajectory(total_frames: int, fps: float) -> list:
    """
    Lissajous (8 şekli) yörünge üretir.
    Her kare için (x_metre, y_metre, z_metre) döndürür.
    Başlangıç noktası (0, 0, ALTITUDE).
    """
    trajectory = []
    for i in range(total_frames):
        t = i / fps  # saniye
        phase = 2.0 * math.pi * t / ORBIT_PERIOD

        # Lissajous eğrisi (8 şekli): x = A*sin(t), y = B*sin(2t)
        x = ORBIT_A * math.sin(phase)
        y = ORBIT_B * math.sin(2.0 * phase)

        # Z ekseni: 35m ile 65m arasında dinamik yükseklik değişimi (tırmanma/alçalma)
        z = ALTITUDE + 15.0 * math.sin(phase * 2.0)
        
        trajectory.append((x, y, z))

    return trajectory


# ─── Kare Üretimi ────────────────────────────────────────────────────── #

def extract_frame(ground_map: np.ndarray, x_m: float, y_m: float,
                  z_m: float) -> np.ndarray:
    """
    Drone konumuna göre zemin haritasından 1920x1080 kırpma yapar.
    Pinhole kamera modeli: GSD = z / fx
    """
    map_h, map_w = ground_map.shape[:2]

    # GSD (Ground Sampling Distance) = altitude / focal_length
    gsd_x = z_m / FX  # metre/piksel
    gsd_y = z_m / FY

    # Kameranın gördüğü alan (metre cinsinden)
    fov_w_m = FRAME_W * gsd_x
    fov_h_m = FRAME_H * gsd_y

    # Drone konumunu harita piksel koordinatına çevir
    # Harita merkezi = (0, 0) metre konumuna karşılık gelir
    center_px = int(map_w / 2.0 + x_m / gsd_x)
    center_py = int(map_h / 2.0 + y_m / gsd_y)

    # Kırpma sınırları (harita piksel koordinatlarında)
    # Kameranın gördüğü piksel sayısı = FRAME boyutu (çünkü GSD zaten ölçekleme)
    half_w = FRAME_W // 2
    half_h = FRAME_H // 2

    x1 = center_px - half_w
    y1 = center_py - half_h
    x2 = center_px + half_w
    y2 = center_py + half_h

    # Sınır kontrolü - padding gerekirse siyah dolgu
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - map_w)
    pad_bottom = max(0, y2 - map_h)

    x1_safe = max(0, x1)
    y1_safe = max(0, y1)
    x2_safe = min(map_w, x2)
    y2_safe = min(map_h, y2)

    crop = ground_map[y1_safe:y2_safe, x1_safe:x2_safe].copy()

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(30, 30, 30),
        )

    # Boyut garantisi
    if crop.shape[0] != FRAME_H or crop.shape[1] != FRAME_W:
        crop = cv2.resize(crop, (FRAME_W, FRAME_H))

    return crop


# ─── Ana Üretim ──────────────────────────────────────────────────────── #

def main():
    start = time.time()
    print("=" * 60)
    print("TEKNOFEST GÖREV 2 - SENTETİK VERİ SETİ ÜRETECİ")
    print("=" * 60)
    print(f"  FPS: {FPS}")
    print(f"  Toplam Süre: {DURATION_SEC}s ({TOTAL_FRAMES} kare)")
    print(f"  GPS Sağlıklı: İlk {GPS_HEALTHY_SEC}s ({GPS_HEALTHY_FRAMES} kare)")
    print(f"  GPS Kesik: Son {DURATION_SEC - GPS_HEALTHY_SEC}s ({TOTAL_FRAMES - GPS_HEALTHY_FRAMES} kare)")
    print(f"  Çözünürlük: {FRAME_W}x{FRAME_H}")
    print(f"  İrtifa: {ALTITUDE}m")
    print("=" * 60)

    # Zemin haritası
    ground_map = generate_ground_map(MAP_W, MAP_H)

    # Zemin haritasını kaydet (debug için)
    map_path = os.path.join(OUTPUT_DIR, "ground_map.png")
    cv2.imwrite(map_path, ground_map)
    print(f"  Zemin haritası kaydedildi: {map_path}")

    # Yörünge hesapla
    print("[2/4] Uçuş yörüngesi hesaplanıyor...")
    trajectory = compute_trajectory(TOTAL_FRAMES, FPS)

    # Kareleri üret
    print("[3/4] Kareler üretiliyor...")
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Landmarkları üret (drone uçuş alanı içinde rastgele)
    # Drone (-40, -25) ile (40, 25) arası dolaşıyor
    for _ in range(NUM_LANDMARKS):
        lx = np.random.uniform(-35, 35)
        ly = np.random.uniform(-20, 20)
        LANDMARKS.append((lx, ly, LANDMARK_SIZE_M, "UAP"))

    ground_truth = {
        "metadata": {
            "fps": FPS,
            "total_frames": TOTAL_FRAMES,
            "gps_healthy_frames": GPS_HEALTHY_FRAMES,
            "duration_sec": DURATION_SEC,
            "altitude_m": ALTITUDE,
            "frame_width": FRAME_W,
            "frame_height": FRAME_H,
            "camera": {
                "fx": FX, "fy": FY,
                "cx": CX, "cy": CY,
                "tilt_deg": 0.0,
            },
            "trajectory_type": "lissajous_figure_8",
            "orbit_a_m": ORBIT_A,
            "orbit_b_m": ORBIT_B,
        },
        "frames": [],
    }

    for i in range(TOTAL_FRAMES):
        x, y, z = trajectory[i]
        health = 1 if i < GPS_HEALTHY_FRAMES else 0

        # Kare üret
        frame = extract_frame(ground_map, x, y, z)

        # Kaydet
        fname = f"frame_{i:05d}.png"
        fpath = os.path.join(frames_dir, fname)
        cv2.imwrite(fpath, frame)

        # Kameranın FOV'undaki landmarkları bul
        detections = []
        for lx, ly, size_m, cls_name in LANDMARKS:
            # Rölatif pozisyon (nadire bakan kamera)
            dx = lx - x
            dy = ly - y

            # Piksele izdüşüm (dinamik z kullanılarak)
            px = (dx / z) * FX + CX
            py = (dy / z) * FY + CY

            # Boyut izdüşümü
            bbox_w = (size_m / z) * FX
            bbox_h = (size_m / z) * FY

            # FOV içinde mi? (100px paylı)
            margin = 100
            if -margin < px < FRAME_W + margin and -margin < py < FRAME_H + margin:
                detections.append({
                    "class_name": cls_name,
                    "confidence": float(np.random.uniform(0.8, 1.0)),
                    "bbox_x": float(px),
                    "bbox_y": float(py),
                    "bbox_w": float(bbox_w),
                    "bbox_h": float(bbox_h)
                })

        # Ground truth kaydı (yarışma JSON formatına uygun)
        frame_entry = {
            "frame_id": i,
            "timestamp": i / FPS,
            "translation_x": round(x, 6),
            "translation_y": round(y, 6),
            "translation_z": round(z, 6),
            "health_status": health,
            "filename": fname,
            "detections": detections,
        }
        ground_truth["frames"].append(frame_entry)

        # İlerleme göstergesi
        if (i + 1) % 100 == 0 or i == 0:
            pct = (i + 1) / TOTAL_FRAMES * 100
            print(f"    [{pct:5.1f}%] Frame {i + 1}/{TOTAL_FRAMES} | "
                  f"Pos=({x:.1f}, {y:.1f}, {z:.1f}) | GPS={'OK' if health else 'KESİK'}")

    # Ground truth kaydet
    print("[4/4] Ground truth kaydediliyor...")
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"✅ VERİ SETİ ÜRETİMİ TAMAMLANDI!")
    print(f"  Süre: {elapsed:.1f}s")
    print(f"  Kareler: {frames_dir}/ ({TOTAL_FRAMES} adet)")
    print(f"  Ground Truth: {gt_path}")
    print(f"  Zemin Haritası: {map_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
