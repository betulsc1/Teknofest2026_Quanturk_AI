"""
src/task1_detection/postprocessor.py

Tespit sonuçlarını temizler ve şartname kurallarını uygular.
"""

import numpy as np
from src.utils.logger import setup_logger
from src.utils.iou_calculator import calculate_iou, nms
from src.utils.bbox_utils import bbox_area, clip_bbox

logger = setup_logger(__name__)

# Şartname Tablo 2: Her sınıf için geçerli motion_status ve landing_status
CLASS_RULES = {
    0: {"motion": [0, 1],  "landing": [-1]},      # Taşıt: ms=0/1, ls=-1
    1: {"motion": [-1],    "landing": [-1]},      # İnsan: ms=-1, ls=-1
    2: {"motion": [-1],    "landing": [0, 1]},    # UAP:   ms=-1, ls=0/1
    3: {"motion": [-1],    "landing": [0, 1]},    # UAİ:   ms=-1, ls=0/1
}

MIN_BBOX_AREA = 16
MIN_BBOX_SIDE = 3

class PostProcessor:
    def __init__(self, iou_threshold: float = 0.45,
                       min_confidence: float = 0.25,
                       max_detections: int = 300):
        self.iou_threshold  = iou_threshold
        self.min_confidence = min_confidence
        self.max_detections = max_detections

    def process(self, detections: list,
                      img_w: int,
                      img_h: int) -> list:
        
        if not detections:
            return []

        detections = self._clip_all(detections, img_w, img_h)
        detections = self._filter_small(detections)
        detections = [d for d in detections if d.get("confidence", 0) >= self.min_confidence]
        detections = self._class_nms(detections)
        detections = self._resolve_rider_conflicts(detections)
        detections = self._enforce_rules(detections)

        detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        detections = detections[:self.max_detections]

        logger.debug(f"PostProcess: {len(detections)} tespit kaldı")
        return detections

    def _clip_all(self, detections: list, img_w: int, img_h: int) -> list:
        for det in detections:
            if "bbox" in det:
                det["bbox"] = clip_bbox(det["bbox"], img_w, img_h)
        return detections

    def _filter_small(self, detections: list) -> list:
        valid = []
        for det in detections:
            if "bbox" not in det:
                continue
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
                continue
            if bbox_area(det["bbox"]) < MIN_BBOX_AREA:
                continue

            valid.append(det)
        return valid

    def _class_nms(self, detections: list) -> list:
        by_class = {}
        for det in detections:
            cls = det.get("class_id", -1)
            by_class.setdefault(cls, []).append(det)

        result = []
        for cls_id, cls_dets in by_class.items():
            kept = nms(cls_dets, iou_threshold=self.iou_threshold)
            result.extend(kept)

        return result

    def _resolve_rider_conflicts(self, detections: list) -> list:
        vehicles = [d for d in detections if d.get("class_id") == 0]
        persons  = [d for d in detections if d.get("class_id") == 1]
        
        to_remove = []

        for v in vehicles:
            if "bbox" not in v: continue
            v_bbox = v["bbox"]
            v_area = bbox_area(v_bbox)

            for p in persons:
                if "bbox" not in p: continue
                p_bbox = p["bbox"]
                p_area = bbox_area(p_bbox)

                ix1 = max(v_bbox[0], p_bbox[0])
                iy1 = max(v_bbox[1], p_bbox[1])
                ix2 = min(v_bbox[2], p_bbox[2])
                iy2 = min(v_bbox[3], p_bbox[3])

                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h

                if inter_area == 0:
                    continue

                min_area = min(v_area, p_area)
                if min_area == 0: continue
                
                overlap_ratio = inter_area / min_area

                if overlap_ratio > 0.60:
                    if v_area >= p_area:
                        if p not in to_remove:
                            to_remove.append(p)
                            logger.debug("Bisiklet/Motosiklet kuralı uygulandı: İnsan silindi.")
                    else:
                        if v not in to_remove:
                            to_remove.append(v)
                            logger.debug("Scooter kuralı uygulandı: Taşıt silindi.")

        return [d for d in detections if d not in to_remove]

    def _enforce_rules(self, detections: list) -> list:
        for det in detections:
            cls = det.get("class_id", -1)
            rules = CLASS_RULES.get(cls, {})

            valid_ms = rules.get("motion", [-1])
            # KESİN ÇÖZÜM: KeyError yememek için .get() ile güvenli okuma yap
            if det.get("motion_status", -1) not in valid_ms:
                det["motion_status"] = valid_ms[0]

            valid_ls = rules.get("landing", [-1])
            if det.get("landing_status", -1) not in valid_ls:
                det["landing_status"] = valid_ls[0]

        return detections