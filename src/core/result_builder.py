"""
src/core/result_builder.py
Her frame işlendikten sonra sunucuya gönderilecek
JSON payload'ını doğrulayarak oluşturur.
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Geçerli değer aralıkları (şartnameye göre)
VALID_CLS            = {"0", "1", "2", "3"}
VALID_LANDING_STATUS = {"-1", "0", "1"}
VALID_MOTION_STATUS  = {"-1", "0", "1"}


class ResultBuilder:
    """
    Pipeline çıktılarını alır, doğrular ve
    sunucunun beklediği JSON formatına dönüştürür.
    """

    def build(self, frame_url: str,
              detections: list,
              position: dict,
              matched_objects: list) -> dict:
        """
        Döndürür: api_client._build_payload'ın anlayacağı dict
        {
          "detections":      [...]
          "position":        {x, y, z}
          "matched_objects": [...]
        }
        """
        clean_detections     = self._build_detections(detections)
        clean_position       = self._build_position(position)
        clean_matched        = self._build_matched(matched_objects)

        return {
            "detections":      clean_detections,
            "position":        clean_position,
            "matched_objects": clean_matched,
        }

    # ------------------------------------------------------------------ #

    def _build_detections(self, detections: list) -> list:
        clean = []
        for det in detections:
            cls_id = str(det.get("class_id", 0))
            ls     = str(det.get("landing_status", -1))
            ms     = str(det.get("motion_status", -1))
            bbox   = det.get("bbox", [0, 0, 1, 1])

            # Şartname kısıtlarına göre ls/ms değerlerini düzelt
            # Taşıt (0): ms=0/1, ls=-1
            # İnsan (1): ms=-1,  ls=-1
            # UAP   (2): ms=-1,  ls=0/1
            # UAİ   (3): ms=-1,  ls=0/1
            cls_id, ms, ls = self._enforce_class_rules(cls_id, ms, ls)

            if cls_id not in VALID_CLS:
                logger.warning(f"Geçersiz sınıf ID atlandı: {cls_id}")
                continue

            clean.append({
                "class_id":       int(cls_id),
                "landing_status": int(ls),
                "motion_status":  int(ms),
                "bbox":           [float(v) for v in bbox],
                "confidence":     float(det.get("confidence", 0.0)),
            })
        return clean

    def _enforce_class_rules(self, cls: str, ms: str,
                              ls: str) -> tuple[str, str, str]:
        """
        Şartnamenin Tablo 2 ve Tablo 4'üne göre değerleri zorla.
        Yanlış değer gönderilirse AP düşer.
        """
        if cls == "1":        # İnsan: ms ve ls her zaman -1
            ms, ls = "-1", "-1"
        elif cls == "0":      # Taşıt: ls her zaman -1
            ls = "-1"
            if ms not in {"0", "1"}:
                ms = "0"
        elif cls in {"2","3"}: # UAP/UAİ: ms her zaman -1
            ms = "-1"
            if ls not in {"0", "1"}:
                ls = "0"
        return cls, ms, ls

    def _build_position(self, position: dict) -> dict:
        return {
            "x": round(float(position.get("x", 0.0)), 4),
            "y": round(float(position.get("y", 0.0)), 4),
            "z": round(float(position.get("z", 0.0)), 4),
        }

    def _build_matched(self, matched_objects: list) -> list:
        clean = []
        for obj in matched_objects:
            clean.append({
                "reference_id": obj.get("reference_id", 0),
                "bbox": [float(v) for v in obj.get("bbox", [0, 0, 1, 1])],
            })
        return clean


# -----------------------------------------------------------------------





