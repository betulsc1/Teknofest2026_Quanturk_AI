"""
src/task1_detection/detector.py

Ana nesne tespit motoru.
YOLOv8m modelini calistirir, 4 sinif tespit eder:
    0 = Tasit
    1 = Insan
    2 = UAP (Ucan Araba Park Alani)
    3 = UAI (Ucan Ambulans Inis Alani)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    _YOLO = None

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


# ------------------------------------------------------------------ #
# Inline yardimci fonksiyonlar
# (src.utils import sorununu onlemek icin burada tanimlandi)
# ------------------------------------------------------------------ #

def _clip_bbox(bbox: list, img_w: int, img_h: int) -> list:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(float(x1), img_w)),
        max(0.0, min(float(y1), img_h)),
        max(0.0, min(float(x2), img_w)),
        max(0.0, min(float(y2), img_h)),
    ]


def _is_thermal(frame: np.ndarray) -> bool:
    if frame is None:
        return False
    if len(frame.shape) == 2:
        return True
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < 30.0


def _thermal_to_rgb(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ #

class ObjectDetector:

    CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.confidence = cfg.get("confidence", 0.35)
        self.iou_thr    = cfg.get("iou_threshold", 0.45)
        self.imgsz      = cfg.get("image_size", 1280)
        self.device     = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_sahi   = False
        self._model     = None
        self._sahi_model = None
        self._sahi_slice_size = 640
        self._sahi_overlap    = 0.2

        self._load_model(cfg.get("model_path", "models/task1/detector/best.pt"))

    def _load_model(self, model_path: str):
        if not YOLO_AVAILABLE:
            logger.error("ultralytics kurulu degil: pip install ultralytics")
            return
        try:
            path = Path(model_path)
            if path.exists():
                logger.info(f"Model yukleniyor: {path}")
                self._model = _YOLO(str(path))
            else:
                logger.warning("Egitilmis model yok, pretrained yolov8m kullaniliyor")
                self._model = _YOLO("yolov8m.pt")
            self._model.to(self.device)
            logger.info(f"Model yuklendi | Cihaz: {self.device}")
        except Exception as e:
            logger.error(f"Model yuklenemedi: {e}")
            self._model = None

    def enable_sahi(self, slice_size: int = 640, overlap: float = 0.2):
        try:
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self._model,
                confidence_threshold=self.confidence,
                device=self.device,
            )
            self._sahi_slice_size = slice_size
            self._sahi_overlap    = overlap
            self.use_sahi         = True
            logger.info(f"SAHI aktif | Dilim: {slice_size}px")
        except ImportError:
            logger.warning("sahi kurulu degil: pip install sahi")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        if frame is None or self._model is None:
            return []
        h, w = frame.shape[:2]
        processed = _thermal_to_rgb(frame) if _is_thermal(frame) else frame
        if self.use_sahi and self._sahi_model is not None:
            return self._detect_sahi(processed, h, w)
        return self._detect_yolo(processed, h, w)

    def _detect_yolo(self, frame: np.ndarray, img_h: int, img_w: int) -> List[Dict]:
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_thr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0], img_h, img_w)

    def _detect_sahi(self, frame: np.ndarray, img_h: int, img_w: int) -> List[Dict]:
        try:
            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                image=frame,
                detection_model=self._sahi_model,
                slice_height=self._sahi_slice_size,
                slice_width=self._sahi_slice_size,
                overlap_height_ratio=self._sahi_overlap,
                overlap_width_ratio=self._sahi_overlap,
                verbose=False,
            )
            detections = []
            for pred in result.object_prediction_list:
                cls_id = int(pred.category.id)
                if cls_id not in self.CLASS_NAMES:
                    continue
                b = pred.bbox
                detections.append({
                    "class_id":       cls_id,
                    "class_name":     self.CLASS_NAMES[cls_id],
                    "confidence":     float(pred.score.value),
                    "bbox":           _clip_bbox([float(b.minx), float(b.miny),
                                                  float(b.maxx), float(b.maxy)], img_w, img_h),
                    "motion_status":  -1,
                    "landing_status": -1,
                })
            return detections
        except Exception as e:
            logger.warning(f"SAHI basarisiz: {e}")
            return self._detect_yolo(frame, img_h, img_w)

    def _parse_results(self, result, img_h: int, img_w: int) -> List[Dict]:
        if result.boxes is None:
            return []
        detections = []
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            if cls_id not in self.CLASS_NAMES:
                continue
            conf = float(result.boxes.conf[i].item())
            xyxy = result.boxes.xyxy[i].cpu().numpy()
            detections.append({
                "class_id":       cls_id,
                "class_name":     self.CLASS_NAMES[cls_id],
                "confidence":     conf,
                "bbox":           _clip_bbox([float(xyxy[0]), float(xyxy[1]),
                                              float(xyxy[2]), float(xyxy[3])], img_w, img_h),
                "motion_status":  -1,
                "landing_status": -1,
            })
        return detections

    def warmup(self):
        logger.info("Model isinıyor...")
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for _ in range(3):
            self._detect_yolo(dummy, 1080, 1920)
        logger.info("Model hazir.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None