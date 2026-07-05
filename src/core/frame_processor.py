"""
src/core/frame_processor.py
Tek bir frame için tüm görevleri sırayla çalıştıran orkestratör.
Görev modüllerini bağlar, onlara bağımlı değildir.
"""

import time
from typing import Optional, List, Dict

import numpy as np

from src.core.result_builder import ResultBuilder
from src.utils.logger import setup_logger
from src.utils.visualizer import compose_debug_frame

logger = setup_logger(__name__)


class FrameProcessor:
    """
    Her frame için şu sırayla çalışır:
    1. Görev 1: Nesne tespiti + hareket + iniş durumu
    2. Görev 2: Pozisyon kestirimi
    3. Görev 3: Referans obje eşleme
    4. Sonuç paketi oluşturma (ResultBuilder)
    """

    def __init__(self, detector, motion_detector, landing_checker,
                 visual_odometry, matcher, debug: bool = False):
        self.detector        = detector
        self.motion_det      = motion_detector
        self.landing_chk     = landing_checker
        self.vo              = visual_odometry
        self.matcher         = matcher
        self.result_builder  = ResultBuilder()
        self.debug           = debug

    def process(self, frame: np.ndarray,
                      prev_frame: Optional[np.ndarray],
                      frame_data: dict,
                      frame_idx: int,
                      total_frames: int) -> dict:
        """
        Tek frame'i işler.

        Döndürür:
        {
          "result"     : ResultBuilder çıktısı (ApiClient'a gönderilecek)
          "debug_frame": görselleştirilmiş frame (debug modda)
          "elapsed_ms" : toplam işlem süresi
        }
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        gps_healthy = frame_data.get("health_status", 1) == 1

        # ---- GÖREV 1: Nesne Tespiti --------------------------------
        detections = self.detector.detect(frame)

        if prev_frame is not None:
            detections = self.motion_det.classify(prev_frame, frame, detections)

        detections = self.landing_chk.check(frame, detections, img_w=w, img_h=h)

        # ---- GÖREV 2: Pozisyon Kestirimi ---------------------------
        # VisualOdometry.process() imzası:
        #   process(frame, frame_data, detections=None) → dict
        # prev_frame VO'nun kendi içinde tutuluyor, dışarıdan gerekmiyor.
        vo_output = self.vo.process(
            frame=frame,
            frame_data=frame_data,
            detections=self._detections_for_landmarks(detections),
        )

        # VO çıktısı geniş ({x,y,z,vx,vy,gps_healthy,...}), ResultBuilder yalnız
        # x/y/z bekliyor. Gerekli olanı çıkarıyoruz.
        position = {
            "x": vo_output.get("x", 0.0),
            "y": vo_output.get("y", 0.0),
            "z": vo_output.get("z", 0.0),
        }

        # ---- GÖREV 3: Görüntü Eşleme -------------------------------
        matched_objects = self.matcher.match_frame(frame)

        # ---- Sonuç paketi ------------------------------------------
        result = self.result_builder.build(
            frame_url=frame_data["url"],
            detections=detections,
            position=position,
            matched_objects=matched_objects,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ---- Debug görselleştirme ----------------------------------
        debug_frame = None
        if self.debug:
            debug_frame = compose_debug_frame(
                frame=frame,
                detections=detections,
                position=position,
                matched_objects=matched_objects,
                frame_idx=frame_idx,
                total=total_frames,
                elapsed_ms=elapsed_ms,
                gps_healthy=gps_healthy,
            )

        return {
            "result":      result,
            "debug_frame": debug_frame,
            "elapsed_ms":  elapsed_ms,
            # Ek bilgiler (logging/debug için)
            "gps_healthy": gps_healthy,
            "det_count":   len(detections),
            "vo_status":   {
                "gps_healthy":      vo_output.get("gps_healthy"),
                "lightglue_active": vo_output.get("lightglue_active"),
                "landmark_active":  vo_output.get("landmark_active"),
                "uncertainty":      vo_output.get("uncertainty"),
            },
        }

    # ------------------------------------------------------------------ #
    #  Yardımcı: dict tespitleri → LandmarkManager.Detection dataclass
    # ------------------------------------------------------------------ #

    def _detections_for_landmarks(self, detections: List[Dict]) -> Optional[List]:
        """
        detector.py tespit formatını (dict, bbox=xyxy) LandmarkManager'ın
        istediği Detection dataclass (bbox=center+wh) formatına çevirir.

        Landmark entegrasyonu hazır değilse None döndürebilir —
        VisualOdometry.process() zaten `detections=None` durumunu handle ediyor.
        """
        if not detections:
            return None

        try:
            # Lazy import — landmark_manager opsiyonel
            from src.task2_position.landmark_manager import Detection as LmDet
        except ImportError:
            return None

        out = []
        for d in detections:
            bbox = d.get("bbox", [0, 0, 1, 1])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            out.append(LmDet(
                class_name=d.get("class_name", "unknown"),
                confidence=float(d.get("confidence", 0.0)),
                bbox_x=(x1 + x2) / 2.0,
                bbox_y=(y1 + y2) / 2.0,
                bbox_w=max(1.0, x2 - x1),
                bbox_h=max(1.0, y2 - y1),
            ))
        return out if out else None