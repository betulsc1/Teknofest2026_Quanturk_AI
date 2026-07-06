"""
src/communication/result_sender.py
JSON sonuç gönderme + rate limit koruması ve Format Dönüştürücü.
"""
import time
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResultSender:
    def __init__(self, api_client, max_retries: int = 3, retry_delay: float = 0.5):
        self.api = api_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._sent_count = 0
        self._failed_count = 0
        self._total_ms = 0.0

    def format_to_teknofest_json(self, frame_url: str, detections: list) -> dict:
        """ Şartname Sayfa 26'daki yapıya zorunlu dönüştürücü[cite: 1] """
        payload = {
            "id": int(time.time()), 
            "user": "http://localhost/users/takiminiz/", # Yarışmada verilecek
            "frame": frame_url,
            "detected_objects": [],
            "detected_translations": [], # 2. Görev için
            "detected_undefined_objects": [] # 3. Görev için
        }
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Şartnamedeki veri türleri string/int dönüşümlerine çok dikkat edilmeli[cite: 1]
            obj_data = {
                "cls": str(det["class_id"]),
                "landing_status": str(det.get("landing_status", -1)),
                "motion_status": str(det.get("motion_status", -1)),
                "top_left_x": float(x1),
                "top_left_y": float(y1),
                "bottom_right_x": float(x2),
                "bottom_right_y": float(y2)
            }
            payload["detected_objects"].append(obj_data)
            
        return payload

    def send(self, frame_url: str, detections: list) -> bool:
        # Önce formata çevir
        teknofest_json = self.format_to_teknofest_json(frame_url, detections)

        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            ok = self.api.send_result(frame_url, teknofest_json)
            elapsed = (time.perf_counter() - t0) * 1000

            if ok:
                self._sent_count += 1
                self._total_ms += elapsed
                return True

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        self._failed_count += 1
        logger.error(f"Gönderim tamamen başarısız: {frame_url}")
        return False
