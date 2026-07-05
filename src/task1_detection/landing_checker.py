"""
src/task1_detection/landing_checker.py

UAP (sınıf 2) ve UAİ (sınıf 3) için iniş durumunu belirler.

Şartname kuralları (Bölüm 2.1.2):
    landing_status = 1 (UYGUN) ANCAK:
        - Alan TAMAMEN kare içinde olmalı (kenar dışına taşmamalı)
        - Alan üzerinde hiçbir nesne (taşıt, insan VEYA TANIMLANAMAYAN NESNE) olmamalı

    landing_status = 0 (UYGUN DEĞİL):
        - Yukarıdaki koşullardan biri sağlanmıyorsa (Perspektif yanılması dahil)
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Kenar toleransı: alana bu kadar piksel kalana kadar "içinde" sayılır
EDGE_MARGIN_PX = 5

# UAP/UAİ üzerindeki nesne alanı oranı eşiği
OVERLAP_AREA_RATIO = 0.0    # %0 — herhangi bir kesişim yeterli


class LandingChecker:
    """
    UAP ve UAİ alanlarının iniş uygunluğunu kontrol eder.
    detector.py çıktısını alır, landing_status doldurur.
    """

    def __init__(self, edge_margin: int = EDGE_MARGIN_PX,
                       overlap_ratio: float = OVERLAP_AREA_RATIO):
        self.edge_margin  = edge_margin
        self.overlap_ratio = overlap_ratio

    # ------------------------------------------------------------------ #
    # Ana fonksiyon
    # ------------------------------------------------------------------ #

    def check(self, frame: np.ndarray,
                detections: List[Dict],
                img_w: Optional[int] = None,
                img_h: Optional[int] = None) -> List[Dict]:
        """
        Her UAP/UAİ tespiti için landing_status atar.

        frame      : mevcut BGR frame (Görsel netlik kontrolü kalktığı için opsiyonel)
        detections : detector.py + motion_detector.py çıktısı
        img_w/h    : görüntü boyutları (None ise frame'den alınır)
        """
        if frame is not None:
            h, w = frame.shape[:2]
        else:
            h = img_h or 1080
            w = img_w or 1920

        if img_w is not None:
            w = img_w
        if img_h is not None:
            h = img_h

        # Sadece UAP ve UAİ alanlarını filtrele
        landing_areas = [d for d in detections if d["class_id"] in (2, 3)]
        
        # DİKKAT: Şartname gereği sadece Taşıt/İnsan değil, UAP/UAİ OLMAYAN HER ŞEY engeldir.
        # Görev 3'teki tanımlanamayan nesneler de bu listeye girmeli.
        other_objects = [d for d in detections if d["class_id"] not in (2, 3)]

        for area in landing_areas:
            bbox = area["bbox"]

            # Kural 1: Alan tamamen kare içinde mi?
            fully_inside = self._is_fully_inside(bbox, w, h)

            if not fully_inside:
                area["landing_status"] = 0
                logger.debug(f"cls{area['class_id']} UYGUN DEĞİL: kenar dışında")
                continue

            # Kural 2: Alan üzerinde nesne var mı? (Genişletilmiş perspektif kontrolü ile)
            has_obstacle = self._has_obstacle(bbox, other_objects)

            if has_obstacle:
                area["landing_status"] = 0
                logger.debug(f"cls{area['class_id']} UYGUN DEĞİL: üzerinde nesne var")
                continue

            # Tüm kontroller geçti
            area["landing_status"] = 1
            logger.debug(f"cls{area['class_id']} UYGUN")

        # UAP/UAİ olmayan nesnelerin landing_status = -1 olmalı
        for det in detections:
            if det["class_id"] not in (2, 3):
                det["landing_status"] = -1

        return detections

    # ------------------------------------------------------------------ #
    # Kural 1: Tamamen kare içinde mi?
    # ------------------------------------------------------------------ #

    def _is_fully_inside(self, bbox: list,
                               img_w: int,
                               img_h: int) -> bool:
        """
        Bbox'ın tamamen görüntü sınırları içinde olup olmadığını kontrol eder.
        Kenar bölgelerine yakın olan alanlar uygun sayılmaz.
        """
        x1, y1, x2, y2 = bbox
        m = self.edge_margin
        return (x1 >= m and
                y1 >= m and
                x2 <= img_w - m and
                y2 <= img_h - m)

    # ------------------------------------------------------------------ #
    # Kural 2: Üzerinde nesne var mı? (Şartname Şekil 11 Perspektif Yanılması)
    # ------------------------------------------------------------------ #

    def _has_obstacle(self, area_bbox: list,
                            objects: List[Dict]) -> bool:
        """
        UAP/UAİ alanı üzerinde engel var mı kontrol eder.
        Perspektif yanılması (Şekil 11) için alan bbox'ı %15 genişletilir.
        """
        ax1, ay1, ax2, ay2 = area_bbox

        perspective_margin = max((ax2 - ax1), (ay2 - ay1)) * 0.15  # %15 margin
        ax1_ext = ax1 - perspective_margin
        ay1_ext = ay1 - perspective_margin
        ax2_ext = ax2 + perspective_margin
        ay2_ext = ay2 + perspective_margin

        for obj in objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]

            # Genişletilmiş UAP alanı ile nesne arasında kesişim var mı?
            ix1 = max(ax1_ext, ox1)
            iy1 = max(ay1_ext, oy1)
            ix2 = min(ax2_ext, ox2)
            iy2 = min(ay2_ext, oy2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            intersection = inter_w * inter_h

            if intersection > self.overlap_ratio:
                logger.debug(
                    f"Engel tespit edildi: cls{obj.get('class_id', 'Bilinmeyen')} "
                    f"(kesişim: {intersection:.0f}px2)"
                )
                return True

        return False

    # ------------------------------------------------------------------ #
    # Yardımcı: UAP ve UAİ bbox büyüklük kontrolü
    # ------------------------------------------------------------------ #

    def validate_area_size(self, bbox: list,
                                  img_w: int,
                                  img_h: int) -> bool:
        """
        Görüntüde çok küçük kalan piksel hatalarını eler. (Opsiyonel)
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        min_side = min(img_w, img_h) * 0.01   # En az %1 boyutunda
        return w >= min_side and h >= min_side