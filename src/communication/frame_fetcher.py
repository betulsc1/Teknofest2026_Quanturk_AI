"""
src/communication/frame_fetcher.py
Frame indirme ve önbellek yönetimi.
api_client.py'yi kullanır, doğrudan requests çağırmaz.
"""

import numpy as np
from typing import Optional
from collections import deque
from src.utils.logger import setup_logger
from src.utils.image_utils import fix_corrupted_frame, is_thermal, thermal_to_rgb

logger = setup_logger(__name__)


class FrameFetcher:
    """
    Sunucudan frame indirmekten sorumlu sınıf.
    - Bozuk frame'leri tespit edip yedeğiyle değiştirir
    - Termal görüntüleri otomatik dönüştürür
    - Son N frame'i bellekte tutar (hareket tespiti için)
    """

    def __init__(self, api_client, buffer_size: int = 5):
        self.api    = api_client
        self.buffer = deque(maxlen=buffer_size)   # son N frame
        self._prev  = None                         # bozuk frame yedekleme

    def fetch(self, frame_data: dict) -> dict:
        """
        Bir frame verisini indirip işleyerek döndürür.

        Döndürür:
        {
          "frame"      : np.ndarray — BGR görüntü
          "is_thermal" : bool
          "raw_frame"  : np.ndarray — orijinal (dönüşümsüz)
          "h"          : int
          "w"          : int
        }
        """
        raw = self.api.fetch_frame_image(frame_data["image_url"])
        raw = fix_corrupted_frame(raw, self._prev)

        thermal = is_thermal(raw)
        frame   = thermal_to_rgb(raw) if thermal else raw

        if thermal:
            logger.debug("Termal görüntü tespit edildi, dönüştürüldü")

        h, w = frame.shape[:2]
        self.buffer.append(frame)
        self._prev = raw

        return {
            "frame":      frame,
            "is_thermal": thermal,
            "raw_frame":  raw,
            "h": h,
            "w": w,
        }

    def previous_frame(self) -> Optional[np.ndarray]:
        """Bir önceki frame'i döndürür (hareket tespiti için)."""
        if len(self.buffer) >= 2:
            return self.buffer[-2]
        return None


