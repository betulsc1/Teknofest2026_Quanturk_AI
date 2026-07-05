"""
src/utils/bbox_utils.py
Bounding box işlemleri — clip, merge, format dönüşümleri.
"""

import numpy as np


def clip_bbox(bbox: list, img_w: int, img_h: int) -> list:
    """
    Bbox koordinatlarının görüntü sınırları dışına çıkmamasını sağlar.
    Şartname: kısmen görünen nesneler de tespit edilmeli.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), img_w))
    y1 = max(0.0, min(float(y1), img_h))
    x2 = max(0.0, min(float(x2), img_w))
    y2 = max(0.0, min(float(y2), img_h))
    return [x1, y1, x2, y2]


def bbox_area(bbox: list) -> float:
    """Bbox alanını hesaplar."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def is_fully_inside(bbox: list, img_w: int, img_h: int,
                    margin: float = 5.0) -> bool:
    """
    Nesnenin tamamının frame içinde olup olmadığını kontrol eder.
    Şartname: UAP/UAI için iniş durumu 'uygun' olabilmesi için
    nesnenin TAMAMEN kare içinde olması gerekir.

    margin: piksel cinsinden kenar toleransı
    """
    x1, y1, x2, y2 = bbox
    return (x1 >= margin and
            y1 >= margin and
            x2 <= img_w - margin and
            y2 <= img_h - margin)


def center_of_bbox(bbox: list) -> tuple[float, float]:
    """Bbox merkez koordinatını döndürür."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def xyxy_to_xywh(bbox: list) -> list:
    """[x1,y1,x2,y2] → [cx,cy,w,h] dönüşümü (YOLO formatı)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return [cx, cy, w, h]


def xywh_to_xyxy(bbox: list) -> list:
    """[cx,cy,w,h] → [x1,y1,x2,y2] dönüşümü."""
    cx, cy, w, h = bbox
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def bboxes_overlap(bbox1: list, bbox2: list) -> bool:
    """İki bbox'ın örtüşüp örtüşmediğini kontrol eder."""
    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2
    return not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a)


def point_in_bbox(point: tuple, bbox: list) -> bool:
    """
    Bir noktanın bbox içinde olup olmadığını kontrol eder.
    UAP/UAI üzerinde nesne var mı kontrolü için kullanılır.
    """
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


# -----------------------------------------------------------------------


