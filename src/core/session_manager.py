"""
src/core/session_manager.py
Oturum yaşam döngüsünü yönetir:
- Frame listesini tutar
- Hangi frame'lerin gönderildiğini takip eder
- GPS sağlık geçişini izler
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SessionManager:
    """
    Bir oturumun (60 dakika, 2250 frame) tüm yaşam döngüsünü yönetir.
    """

    def __init__(self):
        self.frame_list: list  = []
        self.total_frames: int = 0
        self.current_idx: int  = 0

        # GPS geçiş takibi
        self.gps_went_unhealthy: bool  = False
        self.gps_unhealthy_start: int  = -1   # hangi frame'de kesildi

        # Gönderim takibi
        self._sent_urls: set = set()

    def load_frames(self, frame_list: list):
        """Frame listesini yükle ve istatistikleri sıfırla."""
        self.frame_list   = frame_list
        self.total_frames = len(frame_list)
        self.current_idx  = 0
        self._sent_urls   = set()
        logger.info(f"Oturum yüklendi: {self.total_frames} frame")

    def next_frame(self) -> dict | None:
        """Sıradaki frame verisini döndürür, oturum bittiyse None."""
        if self.current_idx >= self.total_frames:
            return None
        fd = self.frame_list[self.current_idx]
        self.current_idx += 1
        return fd

    def mark_sent(self, frame_url: str):
        self._sent_urls.add(frame_url)

    def is_sent(self, frame_url: str) -> bool:
        return frame_url in self._sent_urls

    def update_gps_status(self, frame_data: dict):
        """GPS sağlık değişimini izle ve logla."""
        healthy = frame_data.get("health_status", 1) == 1
        if not healthy and not self.gps_went_unhealthy:
            self.gps_went_unhealthy  = True
            self.gps_unhealthy_start = self.current_idx
            logger.warning(
                f"GPS SAĞLIKSIZ oldu! Frame {self.current_idx}/{self.total_frames} "
                f"— Görsel odometri devreye alındı"
            )

    @property
    def progress(self) -> float:
        """0.0 – 1.0 arası ilerleme."""
        if self.total_frames == 0:
            return 0.0
        return self.current_idx / self.total_frames

    @property
    def remaining(self) -> int:
        return self.total_frames - self.current_idx

    def summary(self) -> dict:
        return {
            "total":              self.total_frames,
            "processed":          self.current_idx,
            "sent":               len(self._sent_urls),
            "gps_failed":         self.gps_went_unhealthy,
            "gps_fail_at_frame":  self.gps_unhealthy_start,
        }


# -----------------------------------------------------------------------