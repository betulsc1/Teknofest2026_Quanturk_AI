import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, cfg: dict):
        self.threshold_px: float = float(cfg.get("threshold_px", 8))
        self.min_bg_points: int = int(cfg.get("min_bg_points", 50))

        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._tracks: Dict[int, dict] = {}
        self._next_track_id = 0

    def classify(self, prev_frame: np.ndarray, curr_frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        # Eğer frame yoksa taşıtları hareketsiz varsay ve geç
        if prev_frame is None or curr_frame is None:
            for det in detections:
                if det["class_id"] == 0:
                    det["motion_status"] = 0
            return detections

        # 1. Önce ID ataması (Tracking) yap
        detections = self.update_tracks(detections)

        # 2. Kamera hareketini (Homografi) hesapla
        H = self._compute_homography(prev_frame, curr_frame)

        # 3. Her taşıt için hareket kontrolü yap
        for det in detections:
            if det["class_id"] != 0: # Taşıt değilse
                det["motion_status"] = -1
                continue
                
            if H is None or det["track_id"] not in self._tracks:
                det["motion_status"] = 0 # Homografi veya track yoksa risksiz olan hareketsizliği seç
                continue
            
            # Aracın BİR ÖNCEKİ merkez noktasını çek
            prev_cx, prev_cy = self._tracks[det["track_id"]]["prev_centroid"]
            
            # Aracın ŞU ANKİ merkez noktasını hesapla
            curr_x1, curr_y1, curr_x2, curr_y2 = det["bbox"]
            curr_cx = float((curr_x1 + curr_x2) / 2.0)
            curr_cy = float((curr_y1 + curr_y2) / 2.0)
            
            # Önceki noktayı H matrisi ile şu anki kareye iz düşür
            point = np.array([[[prev_cx, prev_cy]]], dtype=np.float32)
            expected = cv2.perspectiveTransform(point, H)
            
            if expected is None:
                det["motion_status"] = 0
                continue
                
            ex = float(expected[0][0][0])
            ey = float(expected[0][0][1])
            
            # Beklenen kamera kayması ile aracın gerçek konumu arasındaki mesafe
            distance = float(np.sqrt((curr_cx - ex) ** 2 + (curr_cy - ey) ** 2))
            
            # Eşiği aşıyorsa araç motor gücüyle hareket etmiştir
            det["motion_status"] = 1 if distance > self.threshold_px else 0
            
            # Bir sonraki frame için centroid'i güncelle
            self._tracks[det["track_id"]]["prev_centroid"] = (curr_cx, curr_cy)

        return detections

    # [NOT: _compute_homography ve _iou fonksiyonlarınız MÜKEMMEL, onları olduğu gibi bırakabilirsiniz, değiştirmeye gerek yok]
    # Sadece update_tracks içinde küçük bir ekleme yapıyoruz (prev_centroid ekliyoruz):

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        used_tracks: set = set()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
            
            best_iou, best_track = 0.0, -1

            for track_id, track in self._tracks.items():
                if track_id in used_tracks or track["cls"] != det["class_id"]:
                    continue
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou, best_track = iou, track_id

            if best_iou > 0.3 and best_track >= 0:
                det["track_id"] = best_track
                used_tracks.add(best_track)
                self._tracks[best_track]["bbox"] = det["bbox"]
                self._tracks[best_track]["age"] = 0
                # Centroid güncellenmesini classify içinde yapıyoruz ki prev bozulmasın
            else:
                det["track_id"] = self._next_track_id
                self._tracks[self._next_track_id] = {
                    "bbox": det["bbox"],
                    "cls": det["class_id"],
                    "age": 0,
                    "prev_centroid": (cx, cy) # Centroid'i kaydet
                }
                self._next_track_id += 1

        to_remove = [tid for tid, track in self._tracks.items() if track["age"] > 30]
        for tid in to_remove:
            del self._tracks[tid]
        for track in self._tracks.values():
            track["age"] += 1

        return detections
