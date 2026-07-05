#!/usr/bin/env python3
"""
tests/task2/eval_task2.py

Görev 2 Uçtan Uca Değerlendirme Betiği.

Sentetik veri setini kullanarak VisualOdometry modülünü
yarışma koşullarında test eder ve performans raporları üretir.

Çıktılar (tests/task2/ altına):
    results.json          → Sayısal metrikler
    trajectory_plot.png   → Gerçek vs Tahmin yörünge haritası
    error_over_time.png   → Zamana göre hata grafiği
    axis_comparison.png   → X,Y,Z eksen bazlı karşılaştırma
    summary.txt           → Metin bazlı özet rapor

Kullanım:
    python tests/task2/eval_task2.py
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
import math
import numpy as np
import cv2

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.task2_position.visual_odometry import VisualOdometry
from src.task2_position.landmark_manager import Detection

# ─── Yollar ──────────────────────────────────────────────────────────── #

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(ROOT, "data", "datasets", "task2")
GT_PATH = os.path.join(DATASET_DIR, "ground_truth.json")
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
OUTPUT_DIR = os.path.join(ROOT, "tests", "task2")

# ─── Konfigürasyon ──────────────────────────────────────────────────── #

TASK2_CFG = {
    "dt": 1.0 / 7.5,  # 0.1333...
    "ekf": {
        "q_pos": 0.01,
        "q_vel": 0.5,
        "r_gps": 0.05,
        "r_vo": 0.8,
        "dt": 1.0 / 7.5,
    },
    "keyframe_interval": 50,
    "pose_window_size": 10,
    "drift_threshold": 2.0,
    "use_raft": False,
    "lightglue": {
        "map_interval": 5,
        "min_matches": 10,
    },
}


# ─── Metrik Hesaplama ────────────────────────────────────────────────── #

def compute_rmse(gt_positions, pred_positions):
    """
    Şartname Denklem 2: Ortalama Yarışmacı Hata (MAE of Euclidean distances).
    E = (1/N) * Σ √((x̂-x)² + (ŷ-y)² + (ẑ-z)²)
    NOT: Bu RMSE değil, MAE'dir. Şartname bu formülü kullanıyor.
    """
    gt = np.array(gt_positions)
    pred = np.array(pred_positions)
    diff = gt - pred
    euclidean_per_frame = np.sqrt(np.sum(diff ** 2, axis=1))
    mae = np.mean(euclidean_per_frame)
    return mae


def compute_per_axis_rmse(gt_positions, pred_positions):
    """Her eksen için ayrı RMSE."""
    gt = np.array(gt_positions)
    pred = np.array(pred_positions)
    diff = gt - pred
    rmse_x = np.sqrt(np.mean(diff[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(diff[:, 1] ** 2))
    rmse_z = np.sqrt(np.mean(diff[:, 2] ** 2))
    return rmse_x, rmse_y, rmse_z


def compute_max_error(gt_positions, pred_positions):
    """Maksimum Öklidyen hata."""
    gt = np.array(gt_positions)
    pred = np.array(pred_positions)
    diff = gt - pred
    euclidean = np.sqrt(np.sum(diff ** 2, axis=1))
    return float(np.max(euclidean))


def compute_cumulative_drift(gt_positions, pred_positions):
    """Her kare için kümülatif Öklidyen hata."""
    gt = np.array(gt_positions)
    pred = np.array(pred_positions)
    diff = gt - pred
    return np.sqrt(np.sum(diff ** 2, axis=1))


# ─── Grafik Üretimi ─────────────────────────────────────────────────── #

def plot_trajectory(gt_pos, pred_pos, gps_cutoff_frame, output_path):
    """Gerçek vs Tahmin yörünge çizimi (matplotlib olmadan, OpenCV ile)."""
    gt = np.array(gt_pos)
    pred = np.array(pred_pos)

    # Tüm noktaları kapsayacak ölçek
    all_x = np.concatenate([gt[:, 0], pred[:, 0]])
    all_y = np.concatenate([gt[:, 1], pred[:, 1]])
    margin = 10.0
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    canvas_w, canvas_h = 1200, 900
    plot_x = 100  # sol kenar boşluk
    plot_y = 80   # üst kenar boşluk
    plot_w = canvas_w - 200
    plot_h = canvas_h - 180

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    def to_px(x_m, y_m):
        px = int(plot_x + (x_m - x_min) / (x_max - x_min + 1e-9) * plot_w)
        py = int(plot_y + plot_h - (y_m - y_min) / (y_max - y_min + 1e-9) * plot_h)
        return px, py

    # Izgara
    for i in range(11):
        gy = plot_y + int(i * plot_h / 10)
        cv2.line(canvas, (plot_x, gy), (plot_x + plot_w, gy), (60, 60, 60), 1)
        val = y_max - i * (y_max - y_min) / 10
        cv2.putText(canvas, f"{val:.0f}", (10, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        gx = plot_x + int(i * plot_w / 10)
        cv2.line(canvas, (gx, plot_y), (gx, plot_y + plot_h), (60, 60, 60), 1)
        val = x_min + i * (x_max - x_min) / 10
        cv2.putText(canvas, f"{val:.0f}", (gx - 10, plot_y + plot_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Çerçeve
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h),
                  (100, 100, 100), 1)

    # Ground truth (GPS sağlıklı: yeşil, GPS kesik: yeşil dashed-benzeri)
    for i in range(1, len(gt)):
        p1 = to_px(gt[i - 1, 0], gt[i - 1, 1])
        p2 = to_px(gt[i, 0], gt[i, 1])
        color = (0, 200, 0) if i < gps_cutoff_frame else (0, 120, 0)
        cv2.line(canvas, p1, p2, color, 2)

    # Tahmin (GPS sağlıklı: mavi, GPS kesik: kırmızı)
    for i in range(1, len(pred)):
        p1 = to_px(pred[i - 1, 0], pred[i - 1, 1])
        p2 = to_px(pred[i, 0], pred[i, 1])
        color = (255, 180, 0) if i < gps_cutoff_frame else (0, 80, 255)
        cv2.line(canvas, p1, p2, color, 2)

    # GPS kopma noktası
    cut_pt = to_px(gt[gps_cutoff_frame, 0], gt[gps_cutoff_frame, 1])
    cv2.circle(canvas, cut_pt, 8, (0, 0, 255), -1)
    cv2.putText(canvas, "GPS KOPTU", (cut_pt[0] + 12, cut_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Başlangıç noktası
    start_pt = to_px(gt[0, 0], gt[0, 1])
    cv2.circle(canvas, start_pt, 8, (255, 255, 0), -1)
    cv2.putText(canvas, "BASLANGIC", (start_pt[0] + 12, start_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Lejant
    cv2.putText(canvas, "GERCEK YORUNGE vs TAHMIN", (plot_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    ly = canvas_h - 50
    cv2.line(canvas, (plot_x, ly), (plot_x + 30, ly), (0, 200, 0), 2)
    cv2.putText(canvas, "Gercek (GPS Saglikli)", (plot_x + 40, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    cv2.line(canvas, (plot_x + 250, ly), (plot_x + 280, ly), (0, 120, 0), 2)
    cv2.putText(canvas, "Gercek (GPS Kesik)", (plot_x + 290, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1)
    cv2.line(canvas, (plot_x + 480, ly), (plot_x + 510, ly), (255, 180, 0), 2)
    cv2.putText(canvas, "Tahmin (GPS Saglikli)", (plot_x + 520, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)
    cv2.line(canvas, (plot_x + 730, ly), (plot_x + 760, ly), (0, 80, 255), 2)
    cv2.putText(canvas, "Tahmin (GPS Kesik)", (plot_x + 770, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

    # X/Y etiketleri
    cv2.putText(canvas, "X (metre)", (plot_x + plot_w // 2 - 30, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imwrite(output_path, canvas)


def plot_error_over_time(errors, gps_cutoff_frame, fps, output_path):
    """Zamana göre Öklidyen hata grafiği."""
    n = len(errors)
    canvas_w, canvas_h = 1200, 600
    plot_x, plot_y = 100, 60
    plot_w, plot_h = canvas_w - 150, canvas_h - 140

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    max_err = max(errors) * 1.1 + 0.01
    max_time = n / fps

    def to_px(t, e):
        px = int(plot_x + t / max_time * plot_w)
        py = int(plot_y + plot_h - e / max_err * plot_h)
        return max(0, min(canvas_w - 1, px)), max(0, min(canvas_h - 1, py))

    # Izgara
    for i in range(11):
        gy = plot_y + int(i * plot_h / 10)
        cv2.line(canvas, (plot_x, gy), (plot_x + plot_w, gy), (60, 60, 60), 1)
        val = max_err * (10 - i) / 10
        cv2.putText(canvas, f"{val:.1f}m", (10, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    for i in range(11):
        gx = plot_x + int(i * plot_w / 10)
        cv2.line(canvas, (gx, plot_y), (gx, plot_y + plot_h), (60, 60, 60), 1)
        val = max_time * i / 10
        cv2.putText(canvas, f"{val:.0f}s", (gx - 10, plot_y + plot_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h),
                  (100, 100, 100), 1)

    # GPS kopma çizgisi
    cut_time = gps_cutoff_frame / fps
    cut_px = int(plot_x + cut_time / max_time * plot_w)
    cv2.line(canvas, (cut_px, plot_y), (cut_px, plot_y + plot_h), (0, 0, 255), 2)
    cv2.putText(canvas, "GPS KOPTU", (cut_px + 5, plot_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Hata eğrisi
    for i in range(1, n):
        t1 = (i - 1) / fps
        t2 = i / fps
        p1 = to_px(t1, errors[i - 1])
        p2 = to_px(t2, errors[i])
        color = (0, 255, 255) if i < gps_cutoff_frame else (0, 100, 255)
        cv2.line(canvas, p1, p2, color, 2)

    # Başlık
    cv2.putText(canvas, "OKLIDYEN HATA (metre) vs ZAMAN (saniye)", (plot_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Ortalama hata çizgisi
    mean_err_after = np.mean(errors[gps_cutoff_frame:])
    mean_py = to_px(0, mean_err_after)[1]
    cv2.line(canvas, (plot_x, mean_py), (plot_x + plot_w, mean_py), (100, 255, 100), 1)
    cv2.putText(canvas, f"Ort: {mean_err_after:.2f}m", (plot_x + plot_w - 120, mean_py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    cv2.imwrite(output_path, canvas)


def plot_axis_comparison(gt_pos, pred_pos, gps_cutoff_frame, fps, output_path):
    """X, Y, Z eksen bazlı gerçek vs tahmin karşılaştırma."""
    gt = np.array(gt_pos)
    pred = np.array(pred_pos)
    n = len(gt)

    canvas_w = 1200
    subplot_h = 250
    canvas_h = subplot_h * 3 + 100
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    axis_names = ["X (metre)", "Y (metre)", "Z (metre)"]
    max_time = n / fps

    for ax_idx in range(3):
        gt_vals = gt[:, ax_idx]
        pred_vals = pred[:, ax_idx]

        all_vals = np.concatenate([gt_vals, pred_vals])
        v_min = all_vals.min() - 2
        v_max = all_vals.max() + 2

        plot_x, plot_y = 100, 50 + ax_idx * subplot_h
        plot_w, plot_h = canvas_w - 150, subplot_h - 60

        def to_px(t, v):
            px = int(plot_x + t / max_time * plot_w)
            py = int(plot_y + plot_h - (v - v_min) / (v_max - v_min + 1e-9) * plot_h)
            return max(0, min(canvas_w - 1, px)), max(0, min(canvas_h - 1, py))

        # Çerçeve
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h),
                      (80, 80, 80), 1)

        # GPS kopma
        cut_px = int(plot_x + gps_cutoff_frame / fps / max_time * plot_w)
        cv2.line(canvas, (cut_px, plot_y), (cut_px, plot_y + plot_h), (0, 0, 200), 1)

        # Y ekseni etiketleri
        for i in range(5):
            gy = plot_y + int(i * plot_h / 4)
            val = v_max - i * (v_max - v_min) / 4
            cv2.putText(canvas, f"{val:.0f}", (15, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

        # Çizimler
        for i in range(1, n):
            t1 = (i - 1) / fps
            t2 = i / fps

            p1_gt = to_px(t1, gt_vals[i - 1])
            p2_gt = to_px(t2, gt_vals[i])
            cv2.line(canvas, p1_gt, p2_gt, (0, 200, 0), 2)

            p1_pred = to_px(t1, pred_vals[i - 1])
            p2_pred = to_px(t2, pred_vals[i])
            color = (255, 180, 0) if i < gps_cutoff_frame else (0, 80, 255)
            cv2.line(canvas, p1_pred, p2_pred, color, 1)

        # Eksen adı
        cv2.putText(canvas, axis_names[ax_idx], (plot_x + plot_w + 10, plot_y + plot_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Başlık
    cv2.putText(canvas, "EKSEN BAZLI KARSILASTIRMA: Gercek (yesil) vs Tahmin (mavi/turuncu)",
                (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, canvas)


# ─── Ana Değerlendirme ───────────────────────────────────────────────── #

def main():
    start = time.time()
    print("=" * 65)
    print("  TEKNOFEST GÖREV 2 - UÇTAN UCA DEĞERLENDİRME")
    print("=" * 65)

    # Ground truth yükle
    if not os.path.exists(GT_PATH):
        print(f"HATA: Ground truth bulunamadı: {GT_PATH}")
        print("Önce generate_synthetic_dataset.py çalıştırın.")
        sys.exit(1)

    with open(GT_PATH, "r") as f:
        gt_data = json.load(f)

    metadata = gt_data["metadata"]
    frames = gt_data["frames"]
    fps = metadata["fps"]
    gps_healthy_frames = metadata["gps_healthy_frames"]

    camera_cfg = metadata["camera"]
    camera_cfg["fx"] = camera_cfg.get("fx", 1000.0)
    camera_cfg["fy"] = camera_cfg.get("fy", 1000.0)

    print(f"  Toplam kare: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  GPS Sağlıklı: İlk {gps_healthy_frames} kare")
    print(f"  GPS Kesik: Son {len(frames) - gps_healthy_frames} kare")
    print("-" * 65)

    # VisualOdometry başlat
    vo = VisualOdometry(task2_cfg=TASK2_CFG, camera_cfg=camera_cfg)

    gt_positions = []
    pred_positions = []
    frame_times = []
    processing_times = []

    print("\n  İşleniyor...\n")

    for i, fdata in enumerate(frames):
        fname = fdata["filename"]
        fpath = os.path.join(FRAMES_DIR, fname)

        # Kareyi oku
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"  UYARI: Kare okunamadı: {fpath}")
            continue

        # Detections (Landmark simülasyonu)
        detections = []
        for d in fdata.get("detections", []):
            det = Detection(
                class_name = d["class_name"],
                confidence = d["confidence"],
                bbox_x = d["bbox_x"],
                bbox_y = d["bbox_y"],
                bbox_w = d["bbox_w"],
                bbox_h = d["bbox_h"]
            )
            detections.append(det)

        # frame_data hazırla (yarışma formatı)
        frame_data = {
            "translation_x": fdata["translation_x"],
            "translation_y": fdata["translation_y"],
            "translation_z": fdata["translation_z"],
            "health_status": fdata["health_status"],
        }

        # İşle
        t0 = time.time()
        result = vo.process(frame, frame_data, detections)
        dt = time.time() - t0
        processing_times.append(dt)

        gt_positions.append([
            fdata["translation_x"],
            fdata["translation_y"],
            fdata["translation_z"],
        ])
        pred_positions.append([result["x"], result["y"], result["z"]])
        frame_times.append(fdata["timestamp"])

        # İlerleme
        if (i + 1) % 100 == 0 or i == 0 or i == len(frames) - 1:
            gt_x, gt_y, gt_z = fdata["translation_x"], fdata["translation_y"], fdata["translation_z"]
            pr_x, pr_y, pr_z = result["x"], result["y"], result["z"]
            err = math.sqrt((gt_x - pr_x)**2 + (gt_y - pr_y)**2 + (gt_z - pr_z)**2)
            gps_str = "GPS:OK " if fdata["health_status"] else "GPS:OFF"
            print(f"  [{i+1:5d}/{len(frames)}] {gps_str} | "
                  f"GT=({gt_x:7.2f},{gt_y:7.2f},{gt_z:5.1f}) | "
                  f"PR=({pr_x:7.2f},{pr_y:7.2f},{pr_z:5.1f}) | "
                  f"Hata={err:6.2f}m | {dt*1000:5.1f}ms")

    elapsed = time.time() - start

    # ─── Metrikleri Hesapla ──────────────────────────────────────────── #
    print("\n" + "=" * 65)
    print("  METRİKLER")
    print("=" * 65)

    # Genel
    total_rmse = compute_rmse(gt_positions, pred_positions)
    rmse_x, rmse_y, rmse_z = compute_per_axis_rmse(gt_positions, pred_positions)
    max_err = compute_max_error(gt_positions, pred_positions)
    errors = compute_cumulative_drift(gt_positions, pred_positions)

    # Sadece GPS kesik dönem
    gt_blind = gt_positions[gps_healthy_frames:]
    pred_blind = pred_positions[gps_healthy_frames:]
    blind_rmse = compute_rmse(gt_blind, pred_blind)
    blind_rmse_x, blind_rmse_y, blind_rmse_z = compute_per_axis_rmse(gt_blind, pred_blind)

    # Performans
    avg_ms = np.mean(processing_times) * 1000
    max_ms = np.max(processing_times) * 1000
    effective_fps = 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0

    print(f"  Toplam RMSE       : {total_rmse:.4f} metre")
    print(f"  RMSE (X/Y/Z)      : {rmse_x:.4f} / {rmse_y:.4f} / {rmse_z:.4f}")
    print(f"  Max Hata          : {max_err:.4f} metre")
    print(f"  GPS Kesik RMSE    : {blind_rmse:.4f} metre")
    print(f"  GPS Kesik (X/Y/Z) : {blind_rmse_x:.4f} / {blind_rmse_y:.4f} / {blind_rmse_z:.4f}")
    print(f"  Ort. İşleme Süresi: {avg_ms:.1f} ms/kare")
    print(f"  Max İşleme Süresi : {max_ms:.1f} ms/kare")
    print(f"  Efektif FPS       : {effective_fps:.1f}")
    print(f"  Toplam Süre       : {elapsed:.1f}s")

    # ─── Sonuçları Kaydet ────────────────────────────────────────────── #
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "total_rmse": round(total_rmse, 4),
        "rmse_x": round(rmse_x, 4),
        "rmse_y": round(rmse_y, 4),
        "rmse_z": round(rmse_z, 4),
        "max_error": round(max_err, 4),
        "blind_rmse": round(blind_rmse, 4),
        "blind_rmse_x": round(blind_rmse_x, 4),
        "blind_rmse_y": round(blind_rmse_y, 4),
        "blind_rmse_z": round(blind_rmse_z, 4),
        "avg_processing_ms": round(avg_ms, 1),
        "max_processing_ms": round(max_ms, 1),
        "effective_fps": round(effective_fps, 1),
        "total_elapsed_sec": round(elapsed, 1),
        "total_frames": len(frames),
        "gps_healthy_frames": gps_healthy_frames,
        "gps_blind_frames": len(frames) - gps_healthy_frames,
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {results_path}")

    # ─── Grafikler ───────────────────────────────────────────────────── #
    print("\n  Grafikler üretiliyor...")

    traj_path = os.path.join(OUTPUT_DIR, "trajectory_plot.png")
    plot_trajectory(gt_positions, pred_positions, gps_healthy_frames, traj_path)
    print(f"    → {traj_path}")

    err_path = os.path.join(OUTPUT_DIR, "error_over_time.png")
    plot_error_over_time(errors.tolist(), gps_healthy_frames, fps, err_path)
    print(f"    → {err_path}")

    axis_path = os.path.join(OUTPUT_DIR, "axis_comparison.png")
    plot_axis_comparison(gt_positions, pred_positions, gps_healthy_frames, fps, axis_path)
    print(f"    → {axis_path}")

    # ─── Özet Rapor ──────────────────────────────────────────────────── #
    summary_lines = [
        "=" * 65,
        "TEKNOFEST GÖREV 2 - PERFORMANS RAPORU",
        "=" * 65,
        "",
        f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Veri Seti: Sentetik (Lissajous 8-şekli yörünge)",
        f"Toplam Kare: {len(frames)}",
        f"FPS: {fps}",
        f"GPS Sağlıklı: İlk {gps_healthy_frames} kare ({gps_healthy_frames/fps:.0f}s)",
        f"GPS Kesik: Son {len(frames) - gps_healthy_frames} kare ({(len(frames) - gps_healthy_frames)/fps:.0f}s)",
        "",
        "-" * 65,
        "DOĞRULUK METRİKLERİ",
        "-" * 65,
        f"Toplam RMSE              : {total_rmse:.4f} metre",
        f"RMSE X / Y / Z           : {rmse_x:.4f} / {rmse_y:.4f} / {rmse_z:.4f} metre",
        f"Maksimum Hata            : {max_err:.4f} metre",
        "",
        f"GPS Kesik Dönem RMSE     : {blind_rmse:.4f} metre",
        f"GPS Kesik X / Y / Z      : {blind_rmse_x:.4f} / {blind_rmse_y:.4f} / {blind_rmse_z:.4f} metre",
        "",
        "-" * 65,
        "PERFORMANS METRİKLERİ",
        "-" * 65,
        f"Ortalama İşleme Süresi   : {avg_ms:.1f} ms/kare",
        f"Maksimum İşleme Süresi   : {max_ms:.1f} ms/kare",
        f"Efektif FPS              : {effective_fps:.1f}",
        f"Yarışma FPS Sınırı       : 7.5",
        f"FPS Yeterliliği          : {'✅ YETERLİ' if effective_fps >= 7.5 else '❌ YETERSİZ'}",
        "",
        "-" * 65,
        "YARIŞMA DEĞERLENDİRMESİ (Şartname Denklem 2)",
        "-" * 65,
        f"Ortalama Hata (RMSE)     : {blind_rmse:.4f} metre",
        f"Puanlama Notu: Yarışmada sadece GPS kesik dönemdeki hatalar puanlanır.",
        f"              RMSE ne kadar düşükse puan o kadar yüksektir.",
        "",
        "=" * 65,
    ]

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"    → {summary_path}")

    print("\n" + "=" * 65)
    print(f"  ✅ DEĞERLENDİRME TAMAMLANDI!")
    print(f"  📁 Tüm sonuçlar: {OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
