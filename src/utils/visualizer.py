"""
src/utils/visualizer.py
Debug ve izleme amaçlı görselleştirme.
Yarışmada --debug modunda çalışır, production'da kapalı.
"""

import cv2
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Sınıf renkleri (BGR)
CLASS_COLORS = {
    0: (0,   165, 255),  # Taşıt     → turuncu
    1: (0,   255,   0),  # İnsan     → yeşil
    2: (255,   0,   0),  # UAP       → mavi
    3: (0,    0,  255),  # UAİ       → kırmızı
}

CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

MOTION_LABELS = {-1: "", 0: "DUR", 1: "HAR"}
LANDING_LABELS = {-1: "", 0: "UYGUN_DEGIL", 1: "UYGUN"}


def draw_detections(frame: np.ndarray,
                    detections: list,
                    thickness: int = 2) -> np.ndarray:
    """
    Tespit edilen nesneleri frame üzerine çizer.

    detections: [
        {
          "class_id": int,
          "confidence": float,
          "bbox": [x1, y1, x2, y2],
          "motion_status": int,
          "landing_status": int
        }
    ]
    """
    vis = frame.copy()

    for det in detections:
        cls   = det.get("class_id", 0)
        conf  = det.get("confidence", 0.0)
        bbox  = det["bbox"]
        ms    = det.get("motion_status", -1)
        ls    = det.get("landing_status", -1)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        # Kutu
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Etiket metni
        name    = CLASS_NAMES.get(cls, f"cls{cls}")
        motion  = MOTION_LABELS.get(ms, "")
        landing = LANDING_LABELS.get(ls, "")

        label_parts = [f"{name} {conf:.2f}"]
        if motion:
            label_parts.append(motion)
        if landing:
            label_parts.append(landing)
        label = " | ".join(label_parts)

        # Etiket arkaplanı
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_position(frame: np.ndarray,
                  position: dict,
                  gps_healthy: bool) -> np.ndarray:
    """
    Anlık pozisyon bilgisini frame'in sol üstüne yazar.
    GPS durumunu renkle gösterir (yeşil=sağlıklı, kırmızı=sağlıksız).
    """
    vis = frame.copy()
    x = position.get("x", 0.0)
    y = position.get("y", 0.0)
    z = position.get("z", 0.0)

    color  = (0, 255, 0) if gps_healthy else (0, 0, 255)
    status = "GPS:OK" if gps_healthy else "GPS:FAIL - VO aktif"

    lines = [
        f"{status}",
        f"X: {x:+.3f} m",
        f"Y: {y:+.3f} m",
        f"Z: {z:+.3f} m",
    ]

    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def draw_matches(frame: np.ndarray,
                 matched_objects: list) -> np.ndarray:
    """Eşlenen referans nesneleri mor renkte çizer."""
    vis = frame.copy()
    color = (255, 0, 255)   # mor

    for obj in matched_objects:
        ref_id = obj.get("reference_id", "?")
        bbox   = obj["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"REF:{ref_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis


def draw_frame_info(frame: np.ndarray,
                    frame_idx: int,
                    total: int,
                    elapsed_ms: float) -> np.ndarray:
    """Frame numarası ve işlem süresini sağ alta yazar."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    text = f"Frame {frame_idx+1}/{total} | {elapsed_ms:.1f}ms"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(vis, text, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis


def compose_debug_frame(frame: np.ndarray,
                        detections: list,
                        position: dict,
                        matched_objects: list,
                        frame_idx: int,
                        total: int,
                        elapsed_ms: float,
                        gps_healthy: bool) -> np.ndarray:
    """Tüm debug bilgilerini tek fonksiyonla frame üzerine yazar."""
    vis = draw_detections(frame, detections)
    vis = draw_position(vis, position, gps_healthy)
    vis = draw_matches(vis, matched_objects)
    vis = draw_frame_info(vis, frame_idx, total, elapsed_ms)
    return vis