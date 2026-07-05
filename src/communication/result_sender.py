# -----------------------------------------------------------------------


"""
src/communication/result_sender.py
JSON sonuç gönderme + rate limit koruması.
"""

import time
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultSender:
    """
    Sonuçları sunucuya gönderen ve rate limit'i yöneten sınıf.

    Şartname uyarısı:
    - Her frame için yalnızca 1 sonuç gönderilmeli
    - Limit aşılırsa oturum içinde gönderim kabiliyeti engellenir
    """

    def __init__(self, api_client, max_retries: int = 3,
                 retry_delay: float = 0.5):
        self.api         = api_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Gönderim istatistikleri
        self._sent_count   = 0
        self._failed_count = 0
        self._total_ms     = 0.0

    def send(self, frame_url: str, result: dict) -> bool:
        """
        Sonucu gönderir. Başarısız olursa max_retries kadar tekrar dener.

        frame_url : frame_data["url"]
        result    : pipeline çıktısı
        Döndürür  : True = başarılı
        """
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            ok = self.api.send_result(frame_url, result)
            elapsed = (time.perf_counter() - t0) * 1000

            if ok:
                self._sent_count += 1
                self._total_ms   += elapsed
                logger.debug(f"Gönderildi ({elapsed:.1f}ms) — {frame_url}")
                return True

            if attempt < self.max_retries:
                logger.warning(
                    f"Gönderim başarısız (deneme {attempt}/{self.max_retries}), "
                    f"{self.retry_delay}s bekleniyor..."
                )
                time.sleep(self.retry_delay)

        self._failed_count += 1
        logger.error(f"Gönderim tamamen başarısız: {frame_url}")
        return False

    def stats(self) -> dict:
        """Oturum sonu özeti için istatistikler."""
        total = self._sent_count + self._failed_count
        avg   = (self._total_ms / self._sent_count) if self._sent_count else 0
        return {
            "sent":    self._sent_count,
            "failed":  self._failed_count,
            "total":   total,
            "avg_ms":  round(avg, 2),
        }