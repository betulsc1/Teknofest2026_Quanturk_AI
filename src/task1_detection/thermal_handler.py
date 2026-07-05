"""
src/task1_detection/thermal_handler.py

Termal görüntü ön işleme modülü.
Yarışmada drone termal kamera kullanabilir.
Bu modül termal görüntüyü YOLOv9'un anlayacağı formata çevirir.
"""

import cv2
import numpy as np
from src.utils.logger import setup_logger
from src.utils.image_utils import is_thermal

logger = setup_logger(__name__)


class ThermalHandler:
    """
    Termal görüntü işleme sınıfı.

    Desteklenen senaryolar:
    1. Tek kanallı termal (gri) → 3 kanallı pseudo-RGB
    2. Sahte renkli termal (ironbow, rainbow vb.) → normalize edilmiş RGB
    3. Termal + RGB çift kamera → iki görüntüyü füze et
    """

    def __init__(self, clahe_clip: float = 3.0,
                       clahe_grid: int = 8):
        """
        clahe_clip : CLAHE kontrast sınırı (yüksek = daha fazla kontrast)
        clahe_grid : CLAHE grid boyutu (küçük = daha yerel kontrast)
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid)
        )
        self._last_was_thermal = False

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Ana işleme fonksiyonu.
        Termal ise dönüştür, RGB ise olduğu gibi döndür.

        Giriş : BGR numpy array (H x W x 3) veya gri (H x W)
        Çıktı : BGR numpy array (H x W x 3) — model için hazır
        """
        if frame is None:
            logger.warning("ThermalHandler: None frame geldi")
            return frame

        thermal = is_thermal(frame)
        self._last_was_thermal = thermal

        if not thermal:
            return frame

        logger.debug("Termal görüntü işleniyor...")
        return self._thermal_to_pseudo_rgb(frame)

    def _thermal_to_pseudo_rgb(self, frame: np.ndarray) -> np.ndarray:
        """
        Termal görüntüyü pseudo-RGB'ye çevirir.

        Adımlar:
        1. Tek kanala indir (gri)
        2. CLAHE ile kontrast iyileştir
        3. Normalize et (0-255)
        4. 3 kanallı BGR'ye çevir
        """
        # Tek kanala indir
        if len(frame.shape) == 3:
            # Termal pseudo-renkli → gri (ağırlıklı ortalama yerine max kanal)
            # Neden max? Termal'de en parlak kanal sıcak bölgeleri temsil eder
            gray = np.max(frame, axis=2)
        else:
            gray = frame.copy()

        # Histogram eşitleme — termal görüntülerde kontrast çok düşük olabilir
        gray = self._normalize_thermal(gray)

        # CLAHE: yerel kontrast iyileştirme
        enhanced = self.clahe.apply(gray)

        # Hafif bulanıklaştır — termal gürültüyü azalt
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

        # 3 kanallı BGR'ye çevir
        bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return bgr

    def _normalize_thermal(self, gray: np.ndarray) -> np.ndarray:
        """
        Termal görüntüyü 0-255 aralığına normalize eder.
        Min-max normalizasyonu kullanır.
        """
        min_val = float(gray.min())
        max_val = float(gray.max())

        if max_val - min_val < 1e-6:
            # Tamamen düz görüntü (bozuk kare)
            return gray

        normalized = (gray.astype(np.float32) - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)

    def enhance_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Tespit için ek iyileştirme.
        Normal process() çıktısına ek olarak uygulanır.

        Şartname: insanlar ve araçlar termal görüntüde
        arka plandan ısı farkıyla ayrışır — bunu vurgula.
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Morfolojik işlem: küçük gürültüleri temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Kenar vurgulama (unsharp mask)
        blurred = cv2.GaussianBlur(cleaned, (0, 0), 3)
        sharpened = cv2.addWeighted(cleaned, 1.5, blurred, -0.5, 0)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @property
    def last_was_thermal(self) -> bool:
        """Son işlenen frame termal mıydı?"""
        return self._last_was_thermal