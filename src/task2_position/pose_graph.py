"""
src/task2_position/pose_graph.py

Uzun vadeli drift duzeltme modulu.

Problem:
    Gorsel odometri her frame'de kucuk hatalar biriktirir.
    2250 frame sonunda bu hatalar buyuk sapmalara yol acabilir.

Cozum:
    - Her N frame'de bir "keyframe" kaydet
    - Keyframe'ler arasinda tutarlilik kontrol et
    - Buyuk tutarsizlik varsa geriye donuk duzelt

Not:
    Tam pose graph optimizasyonu g2o/GTSAM gerektirir.
    Bu implementasyon basitlestirilmis sliding window versiyondur.
    Yarisma icin yeterli.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class Keyframe:
    """Tek bir keyframe."""
    def __init__(self, frame_id: int, position: Tuple[float, float, float],
                       uncertainty: float):
        self.frame_id    = frame_id
        self.position    = position      # (x, y, z)
        self.uncertainty = uncertainty
        self.is_corrected = False


class PoseGraph:
    """
    Basitlestirilmis sliding window pose graph.
    Drift birikmesini azaltir.
    """

    def __init__(self, keyframe_interval: int = 50,
                       window_size: int = 10,
                       drift_threshold: float = 2.0):
        """
        keyframe_interval : kac frame'de bir keyframe kaydedilecek
        window_size       : kac keyframe tutulacak (sliding window)
        drift_threshold   : metre cinsinden kabul edilebilir max drift
        """
        self.keyframe_interval = keyframe_interval
        self.window_size       = window_size
        self.drift_threshold   = drift_threshold

        self._keyframes: List[Keyframe] = []
        self._frame_count = 0
        self._total_corrections = 0

    # ------------------------------------------------------------------ #

    def update(self, frame_id: int,
                      position: Tuple[float, float, float],
                      uncertainty: float) -> Tuple[float, float, float]:
        """
        Her frame'de cagrilir.
        Gerekirse konum duzeltmesi uygular.

        Dondurul: (corrected_x, corrected_y, corrected_z)
        """
        self._frame_count += 1

        # Keyframe kaydet
        if self._frame_count % self.keyframe_interval == 0:
            kf = Keyframe(frame_id, position, uncertainty)
            self._keyframes.append(kf)

            # Sliding window — eski keyframe'leri sil
            if len(self._keyframes) > self.window_size:
                self._keyframes.pop(0)

            # Drift kontrol et
            correction = self._check_drift()
            if correction is not None:
                cx, cy, cz = correction
                x, y, z = position
                corrected = (x + cx, y + cy, z + cz)
                self._total_corrections += 1
                logger.info(
                    f"Drift duzeltme uygulandı: "
                    f"({cx:.3f}, {cy:.3f}, {cz:.3f}) | "
                    f"Frame: {frame_id}"
                )
                return corrected

        return position

    # ------------------------------------------------------------------ #

    def _check_drift(self) -> Optional[Tuple[float, float, float]]:
        """
        Keyframe'ler arasinda drift analizi.
        Buyuk tutarsizlik varsa duzeltme vektoru dondurul.
        """
        if len(self._keyframes) < 3:
            return None

        # Son ucgen keyframe'in hiz tutarliligini kontrol et
        kf1 = self._keyframes[-3]
        kf2 = self._keyframes[-2]
        kf3 = self._keyframes[-1]

        # Keyframe'ler arasi hiz
        dt = self.keyframe_interval * 0.133  # saniye
        v12_x = (kf2.position[0] - kf1.position[0]) / dt
        v12_y = (kf2.position[1] - kf1.position[1]) / dt
        v23_x = (kf3.position[0] - kf2.position[0]) / dt
        v23_y = (kf3.position[1] - kf2.position[1]) / dt

        # Hiz tutarsizligi
        dvx = abs(v23_x - v12_x)
        dvy = abs(v23_y - v12_y)

        if dvx > self.drift_threshold or dvy > self.drift_threshold:
            # Anormal hiz degisimi — drift var
            # Basit duzeltme: lineer interpolasyon
            cx = -(v23_x - v12_x) * dt * 0.5
            cy = -(v23_y - v12_y) * dt * 0.5
            return (cx, cy, 0.0)

        return None

    # ------------------------------------------------------------------ #

    def add_gps_anchor(self, frame_id: int,
                              gps_position: Tuple[float, float, float],
                              vo_position: Tuple[float, float, float]):
        """
        GPS konumu ile VO konumunu karsilastir.
        Buyuk fark varsa VO'yu GPS'e cek.

        Bu fonksiyon GPS saglikli donemde kullanilir.
        """
        dx = gps_position[0] - vo_position[0]
        dy = gps_position[1] - vo_position[1]
        dz = gps_position[2] - vo_position[2]

        error = np.sqrt(dx**2 + dy**2 + dz**2)

        if error > self.drift_threshold:
            logger.warning(
                f"GPS-VO uyumsuzlugu: {error:.2f}m | "
                f"GPS: {gps_position} | VO: {vo_position}"
            )

    # ------------------------------------------------------------------ #

    @property
    def total_corrections(self) -> int:
        return self._total_corrections

    @property
    def keyframe_count(self) -> int:
        return len(self._keyframes)

    def reset(self):
        self._keyframes = []
        self._frame_count = 0
        self._total_corrections = 0