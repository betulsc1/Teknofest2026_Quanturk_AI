"""
src/task2_position/visual_odometry.py

Ana gorsel odometri motoru.
Tum Gorev 2 modullerini bir araya getirir.

Akis:
    GPS saglikli (ilk ~450 frame):
        → EKF GPS ile guncellenir
        → LightGlue referans haritasına frame eklenir
        → Landmark haritasına nesne eklenir (Gorev 1 hazir olunca)

    GPS kesilince:
        → Optical Flow ile anlık hareket hesapla
        → LightGlue ile referans haritadan konum tahmin et
        → Landmark'lardan konum tahmin et (Gorev 1 hazir olunca)
        → EKF tüm kaynakları birleştir
        → PoseGraph ile drift düzelt
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List

from src.task2_position.optical_flow import OpticalFlowEstimator
from src.task2_position.kalman_filter import PositionEKF
from src.task2_position.scale_estimator import ScaleEstimator
from src.task2_position.pose_graph import PoseGraph  # type: ignore[import]
from src.task2_position.lightglue_matcher import LightGlueMatcher
from src.task2_position.landmark_manager import LandmarkManager, Detection

logger = logging.getLogger(__name__)


class VisualOdometry:
    """
    Gorev 2 ana motoru.
    frame_processor.py tarafindan cagrilir.
    """

    def __init__(self, task2_cfg: dict, camera_cfg: dict):
        """
        task2_cfg  : config.yaml task2 blogu
        camera_cfg : config.yaml camera_params blogu
        """
        self.cfg        = task2_cfg
        self.camera_cfg = camera_cfg

        # ── Alt modüller ──────────────────────────────────────────── #
        self.flow_estimator  = OpticalFlowEstimator(task2_cfg)
        self.ekf             = PositionEKF(task2_cfg.get("ekf", {}))
        self.scale_estimator = ScaleEstimator(camera_cfg, task2_cfg)
        self.pose_graph      = PoseGraph(
            keyframe_interval = task2_cfg.get("keyframe_interval", 50),
            window_size       = task2_cfg.get("pose_window_size", 10),
            drift_threshold   = task2_cfg.get("drift_threshold", 2.0),
        )
        self.lightglue = LightGlueMatcher(task2_cfg.get("lightglue", {}))
        self.landmarks = LandmarkManager(task2_cfg, camera_cfg)

        # ── Frame geçmişi ─────────────────────────────────────────── #
        self._prev_frame:  Optional[np.ndarray] = None
        self._frame_count  = 0

        # ── GPS durum takibi ──────────────────────────────────────── #
        self._gps_healthy        = True
        self._gps_fail_frame     = -1
        self._last_gps_position  = (0.0, 0.0, 0.0)
        self._last_gps_velocity  = (0.0, 0.0)
        self._gps_healthy_frames = 0

        logger.info("VisualOdometry başlatıldı (LightGlue + Landmark + EKF)")

    # ──────────────────────────────────────────────────────────────── #

    def process(self, frame: np.ndarray,
                       frame_data: dict,
                       detections: Optional[List[Detection]] = None) -> Dict:
        """
        Ana işleme fonksiyonu. Her frame için çağrılır.

        frame_data: {
            "translation_x": float,
            "translation_y": float,
            "translation_z": float,
            "health_status": int,   # 1=sağlıklı, 0=kesik
        }

        detections: Görev 1'den gelen tespitler (hazır olunca dolu gelir)

        Döndürür: {
            "x", "y", "z"         : konum (metre)
            "vx", "vy"            : hız (m/s)
            "gps_healthy"         : bool
            "uncertainty"         : float
            "lightglue_active"    : bool
            "landmark_active"     : bool
            "map_size"            : int
        }
        """
        self._frame_count += 1

        # GPS durumunu güncelle
        health  = int(frame_data.get("health_status", 1))
        gps_x   = float(frame_data.get("translation_x", 0.0))
        gps_y   = float(frame_data.get("translation_y", 0.0))
        gps_z   = float(frame_data.get("translation_z", 0.0))
        altitude = abs(gps_z) if gps_z != 0 else self.scale_estimator.last_altitude

        self._update_gps_status(health)

        # ── EKF tahmin adımı ──────────────────────────────────────── #
        self.ekf.predict(self._gps_healthy)

        # ── GPS sağlıklı fazı ─────────────────────────────────────── #
        if self._gps_healthy:
            self._process_healthy_phase(
                frame, gps_x, gps_y, gps_z, altitude, detections
            )

        # ── GPS kesik fazı ────────────────────────────────────────── #
        else:
            self._process_unhealthy_phase(frame, altitude, detections)

        # ── Optik akış (her zaman) ────────────────────────────────── #
        if self._prev_frame is not None:
            self._process_optical_flow(frame, altitude)

        # ── Konum al + drift düzelt ───────────────────────────────── #
        x, y, z    = self.ekf.get_position()
        vx, vy, vz = self.ekf.get_velocity()

        x, y, z = self.pose_graph.update(
            self._frame_count, (x, y, z), self.ekf.get_uncertainty()
        )

        self._prev_frame = frame.copy()

        result = {
            "x":               x,
            "y":               y,
            "z":               z,
            "vx":              vx,
            "vy":              vy,
            "gps_healthy":     self._gps_healthy,
            "uncertainty":     self.ekf.get_uncertainty(),
            "frame_count":     self._frame_count,
            "lightglue_active": not self._gps_healthy and self.lightglue.is_ready,
            "landmark_active":  not self._gps_healthy and self.landmarks.reliable_landmark_count > 0,
            "map_size":        self.lightglue.map_size,
        }

        logger.debug(
            f"Frame {self._frame_count} | "
            f"({x:.2f}, {y:.2f}, {z:.2f}) | "
            f"GPS: {'OK' if self._gps_healthy else 'KESİK'} | "
            f"Harita: {self.lightglue.map_size} frame"
        )

        return result

    # ──────────────────────────────────────────────────────────────── #

    def _process_healthy_phase(self, frame: np.ndarray,
                                       gps_x: float, gps_y: float, gps_z: float,
                                       altitude: float,
                                       detections: Optional[List[Detection]]):
        """GPS sağlıklı fazda yapılacaklar."""

        # EKF'yi GPS ile güncelle
        if gps_x != 0 or gps_y != 0:
            self.ekf.update_gps(gps_x, gps_y, gps_z)
            self.scale_estimator.update_from_gps(altitude)
            self._last_gps_position = (gps_x, gps_y, gps_z)
            self._gps_healthy_frames += 1

            # Ölçek kalibrasyonu için GPS hızını hesapla
            dt = self.cfg.get("dt", 0.133)
            gps_vel = self.scale_estimator.update_gps_velocity(gps_x, gps_y, dt)
            if gps_vel is not None:
                self._last_gps_velocity = gps_vel

        # PoseGraph GPS anchor ekle
        x_vo, y_vo, z_vo = self.ekf.get_position()
        self.pose_graph.add_gps_anchor(
            self._frame_count,
            (gps_x, gps_y, gps_z),
            (x_vo, y_vo, z_vo)
        )

        # LightGlue referans haritasına ekle
        self.lightglue.add_to_map(frame, gps_x, gps_y, gps_z)

        # Landmark haritasına ekle (Görev 1 hazır olunca dolu gelir)
        if detections:
            self.landmarks.update(detections, gps_x, gps_y, gps_z, self._frame_count)

    # ──────────────────────────────────────────────────────────────── #

    def _process_unhealthy_phase(self, frame: np.ndarray,
                                          altitude: float,
                                          detections: Optional[List[Detection]]):
        """GPS kesik fazda ek konum kaynakları."""

        # ── Dinamik Z (İrtifa) Kestirimi (Barometre veya Görsel) ──
        # Z ekseni için (eğer barometre kesikse veya güvenilmezse) UAP kutu boyutundan irtifa hesapla
        if detections:
            z_ests = []
            for d in detections:
                # Gerçek UAP boyutu 4.5m, fx=1000 varsayımıyla (Pinhole Kamera Denklemi)
                z_est = (1000.0 * 4.5) / max(d.bbox_w, 1.0)
                z_ests.append(z_est)
            if z_ests:
                vis_alt = np.mean(z_ests)
                # Yumuşak geçişli (Hareketli ortalama) Z güncellemesi
                altitude = (altitude * 0.2) + (vis_alt * 0.8)
                
        # EKF Z koordinatını sürekli güncelle (Z ekseninde sürüklenmeyi engeller)
        self.ekf.x[2] = altitude
        
        # LightGlue ile genel harita konumlandırması (Çok ağır olduğu için 30 karede 1 çalıştır)
        if self.lightglue.is_ready and (self.frames_without_gps % 30 == 0):
            est_gps_x = float(self.ekf.x[0])
            est_gps_y = float(self.ekf.x[1])
            lg_result = self.lightglue.estimate_position(frame, est_gps_x, est_gps_y)
            if lg_result is not None:
                lg_x, lg_y, lg_z, lg_conf = lg_result
                gps_noise = 1.0 / (lg_conf + 1e-6)
                gps_noise = min(gps_noise, 5.0)
                
                self.ekf.update_gps(lg_x, lg_y, lg_z, gps_noise=gps_noise)
                self.lightglue.add_to_map(frame, lg_x, lg_y, lg_z)
                logger.debug(
                    f"LightGlue EKF güncellemesi: "
                    f"({lg_x:.2f}, {lg_y:.2f}) conf={lg_conf:.3f}"
                )

        # Landmark ile konum tahmini (Görev 1 hazır olunca aktif)
        if detections and self.landmarks.reliable_landmark_count > 0:
            est_gps_x = float(self.ekf.x[0])
            est_gps_y = float(self.ekf.x[1])
            ekf_cov_x = float(self.ekf.P[0, 0])
            ekf_cov_y = float(self.ekf.P[1, 1])
            
            lm_result = self.landmarks.estimate_from_detections(
                detections, altitude, est_gps_x, est_gps_y, ekf_cov_x, ekf_cov_y, self.frames_without_gps
            )
            if lm_result is not None:
                lm_x, lm_y, lm_z, lm_conf = lm_result
                gps_noise = 1.0 / (lm_conf + 1e-6)
                gps_noise = min(gps_noise, 8.0)
                self.ekf.update_gps(lm_x, lm_y, lm_z, gps_noise=gps_noise)
                logger.debug(
                    f"Landmark EKF güncellemesi: "
                    f"({lm_x:.2f}, {lm_y:.2f}) conf={lm_conf:.3f}"
                )

    # ──────────────────────────────────────────────────────────────── #

    def _process_optical_flow(self, curr_frame: np.ndarray,
                                      altitude: float):
        """Optik akış ile EKF'yi güncelle."""
        if self._prev_frame is None:
            return
        flow = self.flow_estimator.compute(self._prev_frame, curr_frame)

        if flow is None:
            return

        bg_dx, bg_dy, confidence = self.flow_estimator.get_background_flow(flow)

        if confidence < 0.1:
            return

        quality = self.flow_estimator.get_flow_quality(flow)
        dt      = self.cfg.get("dt", 0.133)

        vx, vy = self.scale_estimator.pixels_to_meters(bg_dx, bg_dy, altitude, dt)

        # GPS sağlıklıyken ölçek kalibrasyonu yap
        if self._gps_healthy and self._gps_healthy_frames > 3:
            gps_vx, gps_vy = self._last_gps_velocity
            self.scale_estimator.calibrate_scale(vx, vy, gps_vx, gps_vy)

        self.ekf.update_vo(
            vx, vy,
            quality     = quality * confidence,
            gps_healthy = self._gps_healthy,
        )

    # ──────────────────────────────────────────────────────────────── #

    def _update_gps_status(self, health: int):
        """GPS sağlık durumunu güncelle."""
        was_healthy       = self._gps_healthy
        self._gps_healthy = (health == 1)

        if was_healthy and not self._gps_healthy:
            self._gps_fail_frame = self._frame_count
            logger.warning(
                f"GPS KESİLDİ! Frame: {self._frame_count} | "
                f"Son GPS: {self._last_gps_position} | "
                f"LightGlue harita: {self.lightglue.map_size} frame | "
                f"Landmark: {self.landmarks.reliable_landmark_count} güvenilir"
            )

        if not was_healthy and self._gps_healthy:
            logger.info(f"GPS GERİ DÖNDÜ! Frame: {self._frame_count}")

    # ──────────────────────────────────────────────────────────────── #

    def get_displacement_from_start(self) -> Tuple[float, float, float]:
        """Başlangıçtan toplam yer değiştirme."""
        return self.ekf.get_position()

    def get_status(self) -> dict:
        """Sistem durumu özeti."""
        return {
            "frame_count":      self._frame_count,
            "gps_healthy":      self._gps_healthy,
            "map_size":         self.lightglue.map_size,
            "landmark_count":   self.landmarks.landmark_count,
            "reliable_landmarks": self.landmarks.reliable_landmark_count,
            "pose_corrections": self.pose_graph.total_corrections,
            "ekf_uncertainty":  self.ekf.get_uncertainty(),
            "bias_learned":     self.ekf.bias_learned,
        }

    def reset(self):
        """Yeni oturum için sıfırla."""
        self.ekf.reset()
        self.lightglue.reset()
        self.landmarks.reset()
        self.pose_graph.reset()
        self._prev_frame         = None
        self._frame_count        = 0
        self._gps_healthy        = True
        self._gps_fail_frame     = -1
        self._gps_healthy_frames = 0
        logger.info("VisualOdometry sıfırlandı")

    @property
    def gps_healthy(self) -> bool:
        return self._gps_healthy

    @property
    def frames_without_gps(self) -> int:
        if self._gps_fail_frame < 0 or self._gps_healthy:
            return 0
        return self._frame_count - self._gps_fail_frame