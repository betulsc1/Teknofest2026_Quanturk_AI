"""
src/communication/api_client.py

Yarışma sunucusuyla tüm iletişimi yöneten ana sınıf.
Şartnamenin 8. maddesinde tanımlanan JSON formatını kullanır.
"""

import requests
import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CompetitionAPIClient:
    """
    Yarışma sunucusuyla konuşan tek sınıf.
    Diğer modüller doğrudan requests kullanmaz, hep bu sınıfı kullanır.
    """

    def __init__(self, server_url: str, token: str,
                 timeout: int = 5,
                 cls_as_url: bool = True):
        """
        server_url : "http://192.168.1.100:5000"  (yarışma günü verilir)
        token      : takım token'ı (yarışma günü verilir)
        timeout    : istek başına maksimum bekleme süresi (saniye)
        cls_as_url : Şekil 17'deki gibi URL formatı ("http://.../classes/0/") mu?
                     False ise yalnız sınıf ID string ("0","1","2","3")
                     Yarışma günü kesin format paylaşılınca tek yerden değişir.
        """
        self.server_url = server_url.rstrip("/")
        self.token      = token
        self.timeout    = timeout
        self.cls_as_url = cls_as_url

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        })

        # Her frame için kaç sonuç gönderildi — limit kontrolü için
        # Şartname: aynı kare için birden fazla sonuç gönderilirse
        # ilki geçerli, fazlası engel riski oluşturur
        self._sent_frames: set = set()

    # ------------------------------------------------------------------ #
    #  ADIM 1: Frame listesini al
    # ------------------------------------------------------------------ #

    def get_frame_list(self) -> list:
        """
        Sunucudan oturuma ait tüm frame bilgilerini çeker.
        Her eleman şunları içerir:
          url            → frame'in benzersiz adresi (POST için kullanılır)
          image_url      → görselin indirileceği URL
          translation_x/y/z → GPS pozisyonu (metre)
          health_status  → 1=GPS sağlıklı, 0=GPS sağlıksız
        """
        url = f"{self.server_url}/frames/"
        logger.info(f"Frame listesi alınıyor: {url}")

        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        frames = resp.json()
        logger.info(f"Toplam {len(frames)} frame alındı")
        return frames

    # ------------------------------------------------------------------ #
    #  ADIM 2: Frame görselini indir
    # ------------------------------------------------------------------ #

    def fetch_frame_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        image_url: frame_data["image_url"] değeri
        Döndürür: BGR numpy array (OpenCV formatı) veya None (hata durumunda)
        """
        # Görece URL ise sunucu adresini ekle
        if image_url.startswith("/"):
            full_url = f"{self.server_url}{image_url}"
        else:
            full_url = image_url

        try:
            resp = self.session.get(full_url, timeout=self.timeout)
            resp.raise_for_status()

            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning(f"Görsel decode edilemedi: {full_url}")
                return None

            return frame

        except requests.RequestException as e:
            logger.error(f"Frame indirilemedi: {full_url} — {e}")
            return None

    # ------------------------------------------------------------------ #
    #  ADIM 3: Referans obje görselleri (Görev 3)
    # ------------------------------------------------------------------ #

    def get_reference_objects(self) -> list:
        """
        Şartname §2.3: oturum başında paylaşılan referans nesneler.
        Görev 3 için gerekli.
        """
        try:
            resp = self.session.get(
                f"{self.server_url}/reference-objects/",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            refs = resp.json()
            logger.info(f"{len(refs)} referans nesne alındı")
            return refs
        except Exception as e:
            logger.warning(f"Referans nesneler alınamadı: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  ADIM 4: Sonucu gönder
    # ------------------------------------------------------------------ #

    def send_result(self, frame_url: str, result: dict) -> bool:
        """
        Bir frame için tespit sonuçlarını sunucuya gönderir.

        frame_url : frame_data["url"] değeri
        result    : {
            "detections":      [{class_id, bbox, motion_status, landing_status}]
            "position":        {x, y, z}
            "matched_objects": [{reference_id, bbox}]
        }

        Döndürür: True = başarıyla gönderildi
        """
        # Aynı frame için ikinci kez gönderme (şartname uyarısı)
        if frame_url in self._sent_frames:
            logger.warning(f"Bu frame zaten gönderildi, atlanıyor: {frame_url}")
            return False

        payload = self._build_payload(frame_url, result)

        try:
            resp = self.session.post(
                f"{self.server_url}/results/",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._sent_frames.add(frame_url)
            return True

        except requests.RequestException as e:
            logger.error(f"Sonuç gönderilemedi: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  JSON payload oluşturucu (Şekil 17 formatı)
    # ------------------------------------------------------------------ #

    def _build_payload(self, frame_url: str, result: dict) -> dict:
        """
        Şartnamedeki Şekil 17'de gösterilen JSON formatını oluşturur.

        Örnek (Şekil 17):
        {
          "frame": "http://localhost/frames/4000/",
          "detected_objects": [
            {
              "cls": "http://localhost/classes/1/",
              "landing_status": "-1",
              "motion_status": "-1",
              "top_left_x": 262.87, ...
            }
          ],
          "detected_translations": [...],
          "detected_undefined_objects": [...]
        }
        """
        # Görev 1: Tespit edilen nesneler
        detected_objects = []
        for det in result.get("detections", []):
            cls_id = det["class_id"]
            cls_value = self._format_cls(cls_id)

            detected_objects.append({
                "cls":            cls_value,
                "landing_status": str(det["landing_status"]),
                "motion_status":  str(det["motion_status"]),
                "top_left_x":     round(float(det["bbox"][0]), 2),
                "top_left_y":     round(float(det["bbox"][1]), 2),
                "bottom_right_x": round(float(det["bbox"][2]), 2),
                "bottom_right_y": round(float(det["bbox"][3]), 2),
            })

        # Görev 2: Pozisyon kestirimi
        pos = result.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        detected_translations = [{
            "translation_x": round(float(pos.get("x", 0.0)), 4),
            "translation_y": round(float(pos.get("y", 0.0)), 4),
            "translation_z": round(float(pos.get("z", 0.0)), 4),
        }]

        # Görev 3: Eşlenen tanımsız nesneler
        detected_undefined_objects = []
        for obj in result.get("matched_objects", []):
            detected_undefined_objects.append({
                "object_id":      int(obj["reference_id"]),
                "top_left_x":     round(float(obj["bbox"][0]), 2),
                "top_left_y":     round(float(obj["bbox"][1]), 2),
                "bottom_right_x": round(float(obj["bbox"][2]), 2),
                "bottom_right_y": round(float(obj["bbox"][3]), 2),
            })

        return {
            "frame":                      frame_url,
            "detected_objects":           detected_objects,
            "detected_translations":      detected_translations,
            "detected_undefined_objects": detected_undefined_objects,
        }

    def _format_cls(self, cls_id) -> str:
        """
        Şekil 17'ye göre cls alanı URL formatında.
        Yarışma günü kesinleşince bayrakla değiştirilebilir.
        """
        if self.cls_as_url:
            return f"{self.server_url}/classes/{int(cls_id)}/"
        return str(cls_id)

    def test_connection(self) -> bool:
        """Sunucuya bağlantı testi — test oturumu başında çalıştır."""
        try:
            resp = self.session.get(
                f"{self.server_url}/frames/",
                timeout=self.timeout
            )
            logger.info(f"Bağlantı testi: HTTP {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Bağlantı testi başarısız: {e}")
            return False