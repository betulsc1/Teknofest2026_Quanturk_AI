"""
src/utils/image_utils.py
Görüntü ön işleme araçları.
Termal tespiti, resize, renk dönüşümleri burada.
"""

import cv2
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def is_thermal(frame: np.ndarray) -> bool:
    """
    Görüntünün termal kameradan gelip gelmediğini tespit eder.

    Termal görüntü karakteristikleri:
    - Genellikle tek kanal (gri) veya sahte renkli
    - RGB kanalları birbirine çok yakın (düşük satürasyon)
    - Histogram dağılımı farklı
    """
    if frame is None:
        return False

    if len(frame.shape) == 2:
        return True  # Zaten tek kanal → termal

    # BGR → HSV, satürasyon kanalına bak
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_saturation = hsv[:, :, 1].mean()

    # Termal pseudo-color görüntüler bile düşük satürasyonludur
    return float(mean_saturation) < 30.0


def thermal_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Termal görüntüyü modelin anlayacağı pseudo-RGB'ye çevirir.

    CLAHE ile kontrast iyileştirme → normalize → 3 kanala kopyala
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # CLAHE: gece/termal görüntülerde kontrast iyileştirme
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3 kanallı BGR'ye çevir (YOLO BGR bekler)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return rgb


def resize_for_model(frame: np.ndarray,
                     target_size: int = 1280) -> tuple[np.ndarray, float]:
    """
    Frame'i model giriş boyutuna ölçeklendirir.
    Orijinal en-boy oranını korur, kenarları pad'ler.

    Döndürür:
        resized_frame : hedef boyutlu frame
        scale_factor  : orijinal boyuta dönerken kullanılacak ölçek
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Kare boyuta pad ekle
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    return padded, scale


from typing import Optional

def fix_corrupted_frame(frame: np.ndarray,
                        prev_frame: Optional[np.ndarray]) -> np.ndarray:
    """
    Şartname: görüntü donması, bozulma veya tamamen kayıp olabilir.
    Bozuk frame tespit edilirse önceki frame döndürülür.

    Bozukluk kriterleri:
    - Çok düşük varyans → tamamen siyah veya donmuş kare
    - NaN/Inf piksel değeri
    """
    if frame is None:
        logger.warning("Frame None geldi, önceki frame kullanılıyor")
        return prev_frame if prev_frame is not None else np.zeros((1080, 1920, 3), dtype=np.uint8)

    # NaN/Inf kontrolü
    if not np.isfinite(frame).all():
        logger.warning("Frame'de NaN/Inf tespit edildi")
        return prev_frame if prev_frame is not None else frame

    # Donmuş kare kontrolü (önceki frame ile aynı mı?)
    if prev_frame is not None and frame.shape == prev_frame.shape:
        diff = np.abs(frame.astype(int) - prev_frame.astype(int)).mean()
        if diff < 0.5:
            logger.debug("Donmuş kare tespit edildi (diff < 0.5)")
            # Donmuş frame'i yine de döndür (en azından tespit çalışsın)
            return frame

    # Düşük varyans — tamamen siyah/bozuk
    if frame.var() < 1.0:
        logger.warning("Çok düşük varyans, bozuk frame")
        return prev_frame if prev_frame is not None else frame

    return frame


def scale_bboxes_to_original(bboxes: list,
                              scale: float,
                              orig_h: int,
                              orig_w: int) -> list:
    """
    Model çıktısındaki bbox koordinatlarını orijinal frame boyutuna çevirir.

    bboxes: [[x1, y1, x2, y2], ...] (model boyutunda)
    scale : resize_for_model'den dönen ölçek faktörü
    """
    scaled = []
    for x1, y1, x2, y2 in bboxes:
        x1_orig = min(max(x1 / scale, 0), orig_w)
        y1_orig = min(max(y1 / scale, 0), orig_h)
        x2_orig = min(max(x2 / scale, 0), orig_w)
        y2_orig = min(max(y2 / scale, 0), orig_h)
        scaled.append([x1_orig, y1_orig, x2_orig, y2_orig])
    return scaled