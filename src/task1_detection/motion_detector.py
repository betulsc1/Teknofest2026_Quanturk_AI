import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MotionDetector:

    def __init__(self, cfg: dict):
        self.threshold_px: float = float(cfg.get("threshold_px", 8))
        self.min_bg_points: int = int(cfg.get("min_bg_points", 50))

        # Pylance bazen OpenCV'nin ORB tiplerini göremiyor
        self.orb = cv2.ORB_create(nfeatures=500)  # type: ignore[attr-defined]
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._tracks: Dict[int, dict] = {}
        self._next_track_id = 0

    def classify(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:

        if prev_frame is None or curr_frame is None:
            for det in detections:
                if det["class_id"] == 0:
                    det["motion_status"] = 0
            return detections

        H = self._compute_homography(prev_frame, curr_frame)

        for det in detections:
            if det["class_id"] != 0:
                det["motion_status"] = -1
                continue

            if H is None:
                det["motion_status"] = 0
                continue

            det["motion_status"] = self._is_moving(det["bbox"], H)

        return detections

    def _compute_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Optional[np.ndarray]:

        try:
            orig_h, orig_w = prev_frame.shape[:2]

            if orig_w == 0 or orig_h == 0:
                return None

            scale: float = float(640.0 / float(orig_w))

            new_w: int = 640
            new_h: int = int(orig_h * scale)

            prev_small = cv2.resize(prev_frame, (new_w, new_h))
            curr_small = cv2.resize(curr_frame, (new_w, new_h))

            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

            kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
            kp2, des2 = self.orb.detectAndCompute(curr_gray, None)

            if (
                des1 is None
                or des2 is None
                or len(kp1) < self.min_bg_points
                or len(kp2) < self.min_bg_points
            ):
                return None

            matches = self.matcher.match(des1, des2)

            if len(matches) < 10:
                return None

            matches = sorted(matches, key=lambda m: m.distance)
            good = matches[: min(100, len(matches))]

            pts1 = np.array(
                [[float(kp1[m.queryIdx].pt[0]), float(kp1[m.queryIdx].pt[1])]
                for m in good],
                dtype=np.float32,
            ).reshape(-1, 1, 2)


            pts2 = np.array(
                [[float(kp2[m.trainIdx].pt[0]), float(kp2[m.trainIdx].pt[1])]
                for m in good],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            H_small, mask = cv2.findHomography(
                pts1,
                pts2,
                cv2.RANSAC,
                3.0
            )

            if (
                H_small is None
                or not hasattr(H_small, "shape")
                or H_small.shape != (3, 3)
            ):
                return None

            if mask is not None:
                inlier_ratio = float(mask.sum()) / float(len(mask))

                if inlier_ratio < 0.3:
                    return None

            S = np.array(
                [
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            S_inv = np.array(
                [
                    [1.0 / scale, 0.0, 0.0],
                    [0.0, 1.0 / scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            H_temp = np.dot(S_inv, H_small)
            H_orig = np.dot(H_temp, S).astype(np.float32)

            return H_orig

        except Exception as e:
            logger.debug(f"Homografi atlandı (Hata: {e})")
            return None

    def _is_moving(self, bbox: list, H: np.ndarray) -> int:
        x1, y1, x2, y2 = bbox

        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        point = np.array([[[cx, cy]]], dtype=np.float32)
        expected = cv2.perspectiveTransform(point, H)

        if expected is None:
            return 0

        ex = float(expected[0][0][0])
        ey = float(expected[0][0][1])

        distance = float(
            np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
        )

        return 1 if distance > self.threshold_px else 0

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:

        if not self._tracks:
            for det in detections:
                det["track_id"] = self._next_track_id

                self._tracks[self._next_track_id] = {
                    "bbox": det["bbox"],
                    "cls": det["class_id"],
                    "age": 0,
                }

                self._next_track_id += 1

            return detections

        used_tracks: set = set()

        for det in detections:
            best_iou = 0.0
            best_track = -1

            for track_id, track in self._tracks.items():

                if track_id in used_tracks:
                    continue

                if track["cls"] != det["class_id"]:
                    continue

                iou = self._iou(
                    det["bbox"],
                    track["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id

            if best_iou > 0.3 and best_track >= 0:

                det["track_id"] = best_track
                used_tracks.add(best_track)

                self._tracks[best_track]["bbox"] = det["bbox"]
                self._tracks[best_track]["age"] = 0

            else:

                det["track_id"] = self._next_track_id

                self._tracks[self._next_track_id] = {
                    "bbox": det["bbox"],
                    "cls": det["class_id"],
                    "age": 0,
                }

                self._next_track_id += 1

        to_remove = [
            tid
            for tid, track in self._tracks.items()
            if track["age"] > 30
        ]

        for tid in to_remove:
            del self._tracks[tid]

        for track in self._tracks.values():
            track["age"] += 1

        return detections

    @staticmethod
    def _iou(bbox1: list, bbox2: list) -> float:

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if inter == 0:
            return 0.0

        a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        return inter / (a1 + a2 - inter)