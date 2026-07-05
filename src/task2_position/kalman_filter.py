"""
src/task2_position/kalman_filter.py

Extended Kalman Filter (EKF) pozisyon duzelticisi.

Durum vektoru: [x, y, z, vx, vy, vz]
    x, y, z  : konum (metre)
    vx,vy,vz : hiz (metre/saniye)

Olcum kaynaklari:
    1. GPS    : x, y, z  (saglikli donemde)
    2. Gorsel : dx, dy   (her zaman, VO'dan)

Strateji:
    - GPS saglikli : GPS + VO ile guncelle, VO drift'ini ogren
    - GPS kesilen  : Sadece bias-duzeltilmis VO ile devam et
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PositionEKF:
    """
    6-boyutlu EKF: [x, y, z, vx, vy, vz]
    """

    def __init__(self, cfg: dict):
        """
        cfg: config.yaml task2.ekf blogu

        Ornek:
            cfg = {
                "q_pos": 0.01,
                "q_vel": 0.5,
                "r_gps": 0.05,
                "r_vo": 0.8,
                "dt": 0.133
            }
        """
        self.dt    = cfg.get("dt", 0.133)      # 7.5 FPS → ~133ms
        self.q_pos = cfg.get("q_pos", 0.01)    # Process noise: konum
        self.q_vel = cfg.get("q_vel", 0.5)     # Process noise: hiz
        self.r_gps = cfg.get("r_gps", 0.05)    # GPS measurement noise
        self.r_vo  = cfg.get("r_vo", 0.8)      # VO measurement noise

        # Durum vektoru: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

        # Kovaryans matrisi (baslangicta buyuk belirsizlik)
        self.P = np.eye(6) * 100.0

        # Durum gecis matrisi F (sabit ivme modeli)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # Process noise kovaryans Q
        self.Q = self._build_Q()

        # GPS olcum matrisi (x, y, z gozlemleniyor)
        self.H_gps = np.zeros((3, 6))
        self.H_gps[0, 0] = 1.0
        self.H_gps[1, 1] = 1.0
        self.H_gps[2, 2] = 1.0

        # VO olcum matrisi (dx, dy gozlemleniyor — hiz uzayinda)
        self.H_vo = np.zeros((2, 6))
        self.H_vo[0, 3] = 1.0
        self.H_vo[1, 4] = 1.0

        # GPS → VO drift bias tahmini
        self._vo_bias_x = 0.0
        self._vo_bias_y = 0.0
        self._bias_samples = []
        self._bias_learned = False

        # GPS durum takibi
        self._gps_healthy_frames = 0

        logger.info("EKF baslatildi")

    # ------------------------------------------------------------------ #

    def _build_Q(self) -> np.ndarray:
        """Process noise kovaryans matrisi."""
        Q = np.zeros((6, 6))
        Q[0, 0] = self.q_pos
        Q[1, 1] = self.q_pos
        Q[2, 2] = self.q_pos
        Q[3, 3] = self.q_vel
        Q[4, 4] = self.q_vel
        Q[5, 5] = self.q_vel
        return Q

    # ------------------------------------------------------------------ #

    def predict(self, gps_healthy: bool = True):
        """
        EKF ongorme adimi.
        Onceki durumdan yeni durumu tahmin et.
        gps_healthy: GPS kesikse konum sürec gürültüsünü (process noise) artır
                     böylece EKF landmark ölçümlerine daha çok güvensin.
        """
        self.x = self.F @ self.x
        
        Q = self.Q.copy()
        if not gps_healthy:
            # GPS kesik olduğunda Optik Akış'ın bias'ı konumu hızla bozar.
            # EKF'nin konumdan emin olmaması için q_pos'u artır.
            Q[0, 0] = 5.0
            Q[1, 1] = 5.0

        self.P = self.F @ self.P @ self.F.T + Q

    # ------------------------------------------------------------------ #

    def update_gps(self, gps_x: float, gps_y: float, gps_z: float,
                          gps_noise: Optional[float] = None):
        """
        GPS olcumu ile EKF guncelle.

        gps_x, gps_y, gps_z : GPS konumu (metre)
        gps_noise : GPS measurement noise (None ise varsayilan kullan)
        """
        r = gps_noise if gps_noise is not None else self.r_gps
        R_gps = np.eye(3) * r

        z   = np.array([gps_x, gps_y, gps_z])
        y   = z - self.H_gps @ self.x
        S   = self.H_gps @ self.P @ self.H_gps.T + R_gps
        K   = self.P @ self.H_gps.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        # Joseph form — numerik kararlilik icin
        I_KH  = np.eye(6) - K @ self.H_gps
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gps @ K.T

        self._gps_healthy_frames += 1
        logger.debug(f"GPS guncellendi: ({gps_x:.2f}, {gps_y:.2f}, {gps_z:.2f})")

    # ------------------------------------------------------------------ #

    def update_vo(self, vo_vx: float, vo_vy: float,
                         quality: float = 1.0,
                         gps_healthy: bool = True):
        """
        Gorsel odometri olcumu ile EKF guncelle.

        vo_vx, vo_vy : VO'dan hesaplanan hiz (metre/saniye)
        quality      : akis kalitesi (0-1), measurement noise'u etkiler
        gps_healthy  : GPS saglikli mi?
        """
        # Bias duzeltmesi
        corrected_vx = vo_vx - self._vo_bias_x
        corrected_vy = vo_vy - self._vo_bias_y

        # GPS saglikli donemde bias ogrenme
        if gps_healthy and self._gps_healthy_frames >= 5:
            self._learn_bias(vo_vx, vo_vy)

        # Measurement noise: kaliteye gore ayarla
        noise_factor = self.r_vo / (quality + 1e-6)
        R_vo = np.eye(2) * noise_factor

        z   = np.array([corrected_vx, corrected_vy])
        y   = z - self.H_vo @ self.x
        S   = self.H_vo @ self.P @ self.H_vo.T + R_vo
        K   = self.P @ self.H_vo.T @ np.linalg.inv(S)

        # HACK: Çapraz-kovaryans (cross-covariance) etkisini iptal et.
        # Optik akıştan gelen hatalı hızın konumu (x, y, z) direkt bozmasını engelle.
        # Sadece hızı (vx, vy, vz) güncelle.
        K_mod = K.copy()
        K_mod[0:3, :] = 0.0

        self.x = self.x + K_mod @ y

        I_KH  = np.eye(6) - K_mod @ self.H_vo
        self.P = I_KH @ self.P @ I_KH.T + K_mod @ R_vo @ K_mod.T

        logger.debug(f"VO guncellendi: vx={corrected_vx:.3f}, vy={corrected_vy:.3f}")

    # ------------------------------------------------------------------ #

    def _learn_bias(self, vo_vx: float, vo_vy: float):
        """
        GPS saglikli donemde VO drift bias'ini ogren.

        Fikir: GPS bize gercek hizi veriyor.
               VO ile GPS arasindaki fark = bias.
               Bu bias'i GPS kesildikten sonra VO'dan cikar.
        """
        # EKF'nin GPS'ten ogrendigi gercek hiz
        true_vx = self.x[3]
        true_vy = self.x[4]

        # VO bias = VO olcumu - gercek hiz
        bias_x = vo_vx - true_vx
        bias_y = vo_vy - true_vy

        self._bias_samples.append((bias_x, bias_y))

        # Son 30 ornekten ortalama al
        if len(self._bias_samples) > 30:
            self._bias_samples.pop(0)

        if len(self._bias_samples) >= 5:
            self._vo_bias_x = float(np.mean([s[0] for s in self._bias_samples]))
            self._vo_bias_y = float(np.mean([s[1] for s in self._bias_samples]))
            self._bias_learned = True

        logger.debug(f"VO bias: ({self._vo_bias_x:.4f}, {self._vo_bias_y:.4f})")

    # ------------------------------------------------------------------ #

    def get_position(self) -> Tuple[float, float, float]:
        """Mevcut konum tahminini dondurul (x, y, z) metre."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_velocity(self) -> Tuple[float, float, float]:
        """Mevcut hiz tahminini dondurul (vx, vy, vz) m/s."""
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    def get_uncertainty(self) -> float:
        """Konum belirsizligini dondurul (P matrisinin izi)."""
        return float(np.trace(self.P[:3, :3]))

    def reset(self):
        """EKF'yi sifirla."""
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100.0
        self._bias_samples = []
        self._bias_learned = False
        self._gps_healthy_frames = 0
        logger.info("EKF sifirlandi")

    @property
    def bias_learned(self) -> bool:
        return self._bias_learned

    @property
    def vo_bias(self) -> Tuple[float, float]:
        return self._vo_bias_x, self._vo_bias_y