"""
src/task2_position/scale_estimator.py

Piksel hareketini metre cinsine cevirir.

Formul:
    dx_metre = dx_piksel * (irtifa / focal_length_x)
    dy_metre = dy_piksel * (irtifa / focal_length_y)

Bu formul pin-hole kamera modeline dayanir.
Kamera parametreleri camera_params.yaml'dan okunur.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Varsayilan kamera parametreleri
# Yarisma gununde camera_params.yaml'dan okunacak
DEFAULT_FX = 1000.0   # piksel cinsinden focal length x
DEFAULT_FY = 1000.0   # piksel cinsinden focal length y
DEFAULT_CX = 960.0    # optik merkez x (genellikle width/2)
DEFAULT_CY = 540.0    # optik merkez y (genellikle height/2)

# Varsayilan irtifa (GPS kesildikten sonra son bilinen deger kullanilir)
DEFAULT_ALTITUDE = 50.0   # metre


class ScaleEstimator:
    """
    Optik akis piksel degerlerini metre/saniye'ye cevirir.
    """

    def __init__(self, camera_cfg: dict, task2_cfg: dict):
        """
        camera_cfg : camera_params.yaml icerigi
        task2_cfg  : config.yaml task2 blogu
        """
        self.fx = camera_cfg.get("fx", DEFAULT_FX)
        self.fy = camera_cfg.get("fy", DEFAULT_FY)
        self.cx = camera_cfg.get("cx", DEFAULT_CX)
        self.cy = camera_cfg.get("cy", DEFAULT_CY)

        # Drone kamera açısı (derece) — nadire dogru bakan kamera icin 0
        self.camera_tilt_deg = camera_cfg.get("tilt_deg", 0.0)

        # Irtifa takibi
        self._last_altitude  = DEFAULT_ALTITUDE
        self._altitude_history = []

        # ── Dinamik Ölçek Kalibrasyonu ────────────────────────────── #
        # GPS sağlıklı dönemde optik akıştan hesaplanan hızı
        # gerçek GPS hızıyla karşılaştırıp düzeltme faktörü öğren.
        self._scale_samples_x = []
        self._scale_samples_y = []
        self._scale_factor_x  = 1.0  # Başlangıçta 1:1
        self._scale_factor_y  = 1.0
        self._scale_calibrated = False
        self._prev_gps_x = None
        self._prev_gps_y = None

        logger.info(f"ScaleEstimator | fx={self.fx}, fy={self.fy}")

    # ------------------------------------------------------------------ #

    def pixels_to_meters(self, dx_px: float, dy_px: float,
                                altitude: float,
                                dt: float = 0.133) -> Tuple[float, float]:
        """
        Piksel hareketini metre/saniye'ye cevirir.

        dx_px, dy_px : frame arasi piksel kayma
        altitude     : drone irtifasi (metre)
        dt           : frame suresi (saniye)

        Dondurul: (vx, vy) metre/saniye
        """
        if altitude <= 0:
            altitude = self._last_altitude

        self._last_altitude = altitude
        self._altitude_history.append(altitude)
        if len(self._altitude_history) > 30:
            self._altitude_history.pop(0)

        # Kamera egimi duzeltmesi
        tilt_rad = np.deg2rad(self.camera_tilt_deg)
        effective_altitude = altitude * np.cos(tilt_rad)

        if effective_altitude <= 0:
            effective_altitude = altitude

        # Piksel → metre donusumu
        # GSD (Ground Sampling Distance) = altitude / focal_length
        gsd_x = effective_altitude / self.fx   # metre/piksel
        gsd_y = effective_altitude / self.fy   # metre/piksel

        dx_m = dx_px * gsd_x   # metre
        dy_m = dy_px * gsd_y   # metre

        # Dinamik ölçek kalibrasyonu uygula
        dx_m *= self._scale_factor_x
        dy_m *= self._scale_factor_y

        # Metre → metre/saniye
        vx = dx_m / dt
        vy = dy_m / dt

        logger.debug(
            f"px→m | dx={dx_px:.2f}px → {dx_m:.4f}m → vx={vx:.4f}m/s "
            f"(alt={altitude:.1f}m, gsd={gsd_x:.5f}m/px, sf={self._scale_factor_x:.3f})"
        )

        return vx, vy

    # ------------------------------------------------------------------ #

    def calibrate_scale(self, vo_vx: float, vo_vy: float,
                               gps_vx: float, gps_vy: float):
        """
        GPS sağlıklı dönemde çağrılır.
        Optik akış hızı ile gerçek GPS hızını karşılaştırıp
        ölçek düzeltme faktörü öğren.

        Örnek: Optik akış vx=2.0 m/s diyor ama GPS vx=6.0 m/s → faktör=3.0
        """
        # Çok küçük hızlarda kalibrasyon gürültüe hassas olur, atla
        min_speed = 0.3  # m/s

        if abs(vo_vx) > min_speed and abs(gps_vx) > min_speed:
            ratio_x = gps_vx / vo_vx
            if 0.1 < abs(ratio_x) < 20.0:  # Aşırı değerleri filtrele
                self._scale_samples_x.append(ratio_x)

        if abs(vo_vy) > min_speed and abs(gps_vy) > min_speed:
            ratio_y = gps_vy / vo_vy
            if 0.1 < abs(ratio_y) < 20.0:
                self._scale_samples_y.append(ratio_y)

        # Son 100 örnekten medyan al (outlier'lara dayanıklı)
        if len(self._scale_samples_x) > 100:
            self._scale_samples_x = self._scale_samples_x[-100:]
        if len(self._scale_samples_y) > 100:
            self._scale_samples_y = self._scale_samples_y[-100:]

        if len(self._scale_samples_x) >= 10:
            self._scale_factor_x = float(np.median(self._scale_samples_x))
        if len(self._scale_samples_y) >= 10:
            self._scale_factor_y = float(np.median(self._scale_samples_y))

        if len(self._scale_samples_x) >= 10 and len(self._scale_samples_y) >= 10:
            if not self._scale_calibrated:
                logger.info(
                    f"\u00d6l\u00e7ek kalibrasyonu tamamland\u0131: "
                    f"fx={self._scale_factor_x:.3f}, fy={self._scale_factor_y:.3f}"
                )
                self._scale_calibrated = True

    def update_gps_velocity(self, gps_x: float, gps_y: float, dt: float):
        """
        GPS konumlarından hız hesapla (kalibrasyon için).
        Döndür: (vx, vy) veya None
        """
        if self._prev_gps_x is not None:
            vx = (gps_x - self._prev_gps_x) / dt
            vy = (gps_y - self._prev_gps_y) / dt
            self._prev_gps_x = gps_x
            self._prev_gps_y = gps_y
            return vx, vy
        self._prev_gps_x = gps_x
        self._prev_gps_y = gps_y
        return None

    # ------------------------------------------------------------------ #

    def estimate_altitude_from_flow(self, flow_magnitude: float,
                                           known_speed: float) -> float:
        """
        Eger GPS'ten irtifa gelmiyorsa, bilinen hiz ve akis buyuklugunden
        irtifayi tahmin et.

        Formul: altitude = (flow_magnitude * fx) / (known_speed / dt)
        Bu yaklasimldir, sadece GPS tamamen kesildikten sonra kullanilir.
        """
        if known_speed <= 0 or flow_magnitude <= 0:
            return self._last_altitude

        estimated = (flow_magnitude * self.fx) / known_speed
        return float(estimated)

    # ------------------------------------------------------------------ #

    def update_from_gps(self, gps_altitude: float):
        """GPS'ten gelen irtifayi guncelle."""
        if gps_altitude > 0:
            self._last_altitude = gps_altitude

    # ------------------------------------------------------------------ #

    @property
    def last_altitude(self) -> float:
        return self._last_altitude

    @property
    def smooth_altitude(self) -> float:
        """Son 10 frame'in ortalama irtifasi — gurultu azaltma."""
        if not self._altitude_history:
            return self._last_altitude
        recent = self._altitude_history[-10:]
        return float(np.mean(recent))

    @property
    def scale_calibrated(self) -> bool:
        return self._scale_calibrated

    @property
    def scale_factors(self) -> tuple:
        return self._scale_factor_x, self._scale_factor_y