"""
src/task2_position/landmark_manager.py

Görev 1 tespitlerini Görev 2 için landmark'a dönüştürür.

Konsept:
    - GPS sağlıklı fazda tespit edilen nesneler (taşıt, insan, UAP, UAİ)
      GPS koordinatlarıyla birlikte haritaya kaydedilir
    - GPS kesilince bu nesneler tekrar tespit edilirse
      drone'un konumunu tahmin etmek için kullanılır

Örnek:
    GPS fazında: "Frame 100'de şu koordinatta bir taşıt var"
    GPS sonrası: Aynı taşıt yeni frame'de görülüyor
                 → drone o taşıta göre nerede olmalı?
                 → koordinat tahmin et

NOT: Görev 1 eğitimi bitince bu dosya detector.py ile entegre edilecek.
     Şu an placeholder olarak çalışıyor.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Landmark:
    """Haritadaki tek bir nesne landmark'ı."""
    landmark_id:  int
    class_name:   str           # tasit, insan, UAP, UAI
    gps_x:        float
    gps_y:        float
    gps_z:        float
    pixel_x:      float         # ilk tespit edildiğindeki piksel konumu
    pixel_y:      float
    bbox_w:       float         # bounding box genişliği
    bbox_h:       float         # bounding box yüksekliği
    confidence:   float         # tespit güveni
    seen_count:   int = 1       # kaç frame'de görüldü
    last_frame:   int = 0


@dataclass
class Detection:
    """
    Görev 1'den gelen tek tespit.
    detector.py'ın döndürdüğü format.
    """
    class_name:  str
    confidence:  float
    bbox_x:      float   # merkez x (piksel)
    bbox_y:      float   # merkez y (piksel)
    bbox_w:      float
    bbox_h:      float


# ─────────────────────────────────────────────────────────────────────────── #

class LandmarkManager:
    """
    Görev 1 tespitlerini GPS koordinatlı landmark'lara dönüştürür.

    Kullanım:
        # GPS sağlıklı fazda
        manager.update(detections, gps_x, gps_y, gps_z, frame_id)

        # GPS kesilince
        position = manager.estimate_from_detections(detections, camera_cfg)
    """

    MATCH_DISTANCE_THRESHOLD = 50.0   # piksel cinsinden, aynı nesne eşleştirme
    MIN_SEEN_COUNT           = 3      # En az bu kadar görülmüş olmalı
    MAX_LANDMARKS            = 500    # Haritada max landmark sayısı

    def __init__(self, cfg: dict, camera_cfg: dict):
        """
        cfg:        config.yaml task2 blogu
        camera_cfg: config.yaml camera_params blogu
        """
        self.cfg        = cfg
        self.camera_cfg = camera_cfg

        self._landmarks:    Dict[int, Landmark] = {}
        self._next_id       = 0
        self._frame_count   = 0

        # Kamera parametreleri (koordinat dönüşümü için)
        self._focal_length = camera_cfg.get("focal_length", 1000.0)
        self._cx           = camera_cfg.get("cx", 960.0)
        self._cy           = camera_cfg.get("cy", 540.0)

        logger.info("LandmarkManager başlatıldı")

    # ────────────────────────────────────────────────────────────── #

    def update(self, detections: List[Detection],
                      gps_x: float, gps_y: float, gps_z: float,
                      frame_id: int):
        """
        GPS sağlıklı fazda çağrılır.
        Tespitleri mevcut landmark'larla eşleştir veya yeni landmark ekle.
        """
        self._frame_count = frame_id

        if not detections:
            return

        for det in detections:
            # Mevcut landmark'larla eşleştir (dünya koordinatlarında)
            matched_id = self._find_matching_landmark(det, gps_x, gps_y, gps_z)

            if matched_id is not None:
                # Var olan landmark'ı güncelle
                lm = self._landmarks[matched_id]
                lm.seen_count += 1
                lm.last_frame  = frame_id
                lm.confidence  = max(lm.confidence, det.confidence)
                
                # Yeni görüldüğü açıya göre konumunu bir miktar düzelt (hareketli ortalama)
                altitude = abs(gps_z) if abs(gps_z) > 0.1 else 50.0
                scale = altitude / self._focal_length
                dx = (det.bbox_x - self._cx) * scale
                dy = (det.bbox_y - self._cy) * scale
                new_gps_x = gps_x + dx
                new_gps_y = gps_y + dy
                
                lm.gps_x = (lm.gps_x * 0.8) + (new_gps_x * 0.2)
                lm.gps_y = (lm.gps_y * 0.8) + (new_gps_y * 0.2)
                
                lm.pixel_x = det.bbox_x
                lm.pixel_y = det.bbox_y
            else:
                # Yeni landmark ekle
                if len(self._landmarks) < self.MAX_LANDMARKS:
                    self._add_landmark(det, gps_x, gps_y, gps_z, frame_id)

        logger.debug(
            f"Frame {frame_id}: {len(detections)} tespit, "
            f"{len(self._landmarks)} toplam landmark"
        )

    # ────────────────────────────────────────────────────────────── #

    def estimate_from_detections(self,
                                  detections: List[Detection],
                                  altitude: float,
                                  est_gps_x: float,
                                  est_gps_y: float,
                                  ekf_cov_x: float = 25.0,
                                  ekf_cov_y: float = 25.0,
                                  frames_without_gps: int = 0) -> Optional[Tuple[float, float, float, float]]:
        """
        GPS kesilince çağrılır.
        Gelen tespitleri haritadaki landmark'larla eşleştir, konum tahmin et.
        """
        if not detections or not self._landmarks:
            return None

        # Güvenilir landmark'ları filtrele
        reliable = {
            lid: lm for lid, lm in self._landmarks.items()
            if lm.seen_count >= self.MIN_SEEN_COUNT
        }

        if not reliable:
            return None

        matched_positions = []

        for det in detections:
            # GPS kesikse EKF tahminiyle en iyi eşleşmeyi bul
            # Dinamik eşik 1: EKF 3-sigma (EKF kendine ne kadar güveniyor?)
            sigma = np.sqrt(max(ekf_cov_x, ekf_cov_y))
            
            # Dinamik eşik 2: Fiziksel zaman tabanlı sürüklenme payı (drift allowance)
            # Optik akış saniyede maks 2 metre kayabilir varsayımı: (FPS=7.5 -> dt~0.133s)
            time_without_gps = frames_without_gps * 0.133
            drift_allowance = 15.0 + (time_without_gps * 2.0)
            
            # Tolerans, hem istatistiksel hem fiziksel limitleri kapsamalıdır.
            dynamic_threshold = max(15.0, min(150.0, max(sigma * 3.0, drift_allowance)))

            best_landmark = self._find_best_landmark_match(
                det, reliable, est_gps_x, est_gps_y, altitude, dynamic_threshold
            )

            if best_landmark is not None:
                # Landmark koordinatından drone konumunu tersine hesapla
                est_x, est_y = self._reverse_project(det, best_landmark, altitude)
                conf = det.confidence * best_landmark.confidence
                matched_positions.append((est_x, est_y, best_landmark.gps_z, conf))

        if not matched_positions:
            return None

        # Ağırlıklı ortalama
        total_conf = sum(p[3] for p in matched_positions)
        if total_conf < 1e-6:
            return None

        x = sum(p[0] * p[3] for p in matched_positions) / total_conf
        y = sum(p[1] * p[3] for p in matched_positions) / total_conf
        z = sum(p[2] * p[3] for p in matched_positions) / total_conf
        avg_conf = total_conf / len(matched_positions)

        logger.info(
            f"Landmark pozisyon tahmini: ({x:.2f}, {y:.2f}, {z:.2f}) | "
            f"conf={avg_conf:.3f} | {len(matched_positions)} landmark eşleşti"
        )

        return float(x), float(y), float(z), float(min(avg_conf, 1.0))

    # ────────────────────────────────────────────────────────────── #

    def _add_landmark(self, det: Detection,
                             gps_x: float, gps_y: float, gps_z: float,
                             frame_id: int):
        """Yeni landmark oluştur ve haritaya ekle."""
        altitude = abs(gps_z) if abs(gps_z) > 0.1 else 50.0
        scale = altitude / self._focal_length
        dx = (det.bbox_x - self._cx) * scale
        dy = (det.bbox_y - self._cy) * scale
        
        lm_gps_x = gps_x + dx
        lm_gps_y = gps_y + dy

        lm = Landmark(
            landmark_id = self._next_id,
            class_name  = det.class_name,
            gps_x       = lm_gps_x,
            gps_y       = lm_gps_y,
            gps_z       = gps_z,
            pixel_x     = det.bbox_x,
            pixel_y     = det.bbox_y,
            bbox_w      = det.bbox_w,
            bbox_h      = det.bbox_h,
            confidence  = det.confidence,
            last_frame  = frame_id,
        )
        self._landmarks[self._next_id] = lm
        self._next_id += 1

    def _find_matching_landmark(self, det: Detection,
                                       gps_x: float, gps_y: float, gps_z: float) -> Optional[int]:
        """
        Tespit ile en yakın landmark'ı bul (Dünya koordinatlarında).
        """
        best_id   = None
        best_dist = 15.0  # 15 metre tolerans

        altitude = abs(gps_z) if abs(gps_z) > 0.1 else 50.0
        scale = altitude / self._focal_length
        dx = (det.bbox_x - self._cx) * scale
        dy = (det.bbox_y - self._cy) * scale
        det_gps_x = gps_x + dx
        det_gps_y = gps_y + dy

        for lid, lm in self._landmarks.items():
            if lm.class_name != det.class_name:
                continue

            dist = np.sqrt((lm.gps_x - det_gps_x) ** 2 + (lm.gps_y - det_gps_y) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_id   = lid

        return best_id

    def _find_best_landmark_match(self, det: Detection,
                                         landmarks: Dict[int, Landmark],
                                         est_gps_x: float, est_gps_y: float, altitude: float,
                                         threshold: float) -> Optional[Landmark]:
        """GPS kesildikten sonra tespit ile en iyi landmark eşleştir (Dinamik tolerans ile)."""
        best_lm   = None
        best_dist = threshold

        scale = altitude / self._focal_length
        dx = (det.bbox_x - self._cx) * scale
        dy = (det.bbox_y - self._cy) * scale
        det_gps_x = est_gps_x + dx
        det_gps_y = est_gps_y + dy

        for lm in landmarks.values():
            if lm.class_name != det.class_name:
                continue

            dist = np.sqrt((lm.gps_x - det_gps_x) ** 2 + (lm.gps_y - det_gps_y) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_lm   = lm

        return best_lm

    def _reverse_project(self, det: Detection,
                                lm: Landmark,
                                altitude: float) -> Tuple[float, float]:
        """
        Landmark'ın bilinen dünya koordinatı (GPS) ve
        ekrandaki piksel konumundan drone'un olması gereken konumu bul.
        """
        scale = altitude / self._focal_length
        
        dx = (det.bbox_x - self._cx) * scale
        dy = (det.bbox_y - self._cy) * scale
        
        drone_x = lm.gps_x - dx
        drone_y = lm.gps_y - dy
        
        return drone_x, drone_y

    # ────────────────────────────────────────────────────────────── #

    @property
    def landmark_count(self) -> int:
        return len(self._landmarks)

    @property
    def reliable_landmark_count(self) -> int:
        return sum(1 for lm in self._landmarks.values()
                   if lm.seen_count >= self.MIN_SEEN_COUNT)

    def get_stats(self) -> dict:
        """İstatistikler (debug için)."""
        class_counts = {}
        for lm in self._landmarks.values():
            class_counts[lm.class_name] = class_counts.get(lm.class_name, 0) + 1

        return {
            "total":      len(self._landmarks),
            "reliable":   self.reliable_landmark_count,
            "by_class":   class_counts,
        }

    def reset(self):
        """Yeni oturum için sıfırla."""
        self._landmarks.clear()
        self._next_id    = 0
        self._frame_count = 0
        logger.info("LandmarkManager sıfırlandı")