"""
src/task1_detection/landing_checker.py
UAP ve UAİ iniş durumunu belirler.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

EDGE_MARGIN_PX = 5
OVERLAP_AREA_RATIO = 0.0  # Şartname gereği, en ufak kesişim bile engeli tetikler

class LandingChecker:
    def __init__(self, edge_margin: int = EDGE_MARGIN_PX, overlap_ratio: float = OVERLAP_AREA_RATIO):
        self.edge_margin = edge_margin
        self.overlap_ratio = overlap_ratio

    def check(self, frame: np.ndarray, detections: List[Dict], img_w: Optional[int] = None, img_h: Optional[int] = None) -> List[Dict]:
        # Çözünürlük kesinlikle dinamik çekilmeli
        if frame is not None:
            h, w = frame.shape[:2]
        elif img_w is not None and img_h is not None:
            w, h = img_w, img_h
        else:
            logger.error("Görüntü boyutları algılanamadı, varsayılan 1920x1080 kullanılıyor.")
            w, h = 1920, 1080

        landing_areas = [d for d in detections if d["class_id"] in (2, 3)]
        other_objects = [d for d in detections if d["class_id"] not in (2, 3)]

        for area in landing_areas:
            bbox = area["bbox"]
            
            # Kural 1: Alan kare içinde mi?
            if not self._is_fully_inside(bbox, w, h):
                area["landing_status"] = 0
                continue
                
            # Kural 2: Üzerinde nesne var mı? (Şekil 11 perspektif uyarısı[cite: 1])
            if self._has_obstacle(bbox, other_objects):
                area["landing_status"] = 0
                continue
                
            area["landing_status"] = 1

        for det in detections:
            if det["class_id"] not in (2, 3):
                det["landing_status"] = -1

        return detections

    def _is_fully_inside(self, bbox: list, img_w: int, img_h: int) -> bool:
        x1, y1, x2, y2 = bbox
        m = self.edge_margin
        return (x1 >= m and y1 >= m and x2 <= img_w - m and y2 <= img_h - m)

    def _has_obstacle(self, area_bbox: list, objects: List[Dict]) -> bool:
        ax1, ay1, ax2, ay2 = area_bbox
        perspective_margin = max((ax2 - ax1), (ay2 - ay1)) * 0.15
        
        ax1_ext, ay1_ext = ax1 - perspective_margin, ay1 - perspective_margin
        ax2_ext, ay2_ext = ax2 + perspective_margin, ay2 + perspective_margin

        for obj in objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            ix1 = max(ax1_ext, ox1)
            iy1 = max(ay1_ext, oy1)
            ix2 = min(ax2_ext, ox2)
            iy2 = min(ay2_ext, oy2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            intersection = inter_w * inter_h

            if intersection > self.overlap_ratio:
                return True
        return False
