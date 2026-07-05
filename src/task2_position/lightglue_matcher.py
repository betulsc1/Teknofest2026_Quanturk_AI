"""
src/task2_position/lightglue_matcher.py

LightGlue tabanlı görüntü eşleştirme ve referans harita yönetimi.

Görev:
    1. GPS sağlıklı fazda (ilk 450 frame): 
       - Her frame'den SuperPoint feature'ları çıkar
       - Frame + GPS koordinatını referans haritaya ekle
    
    2. GPS kesilince:
       - Yeni frame'i haritadaki en benzer frame'lerle eşleştir
       - Eşleşen frame'lerin koordinatından drone konumunu tahmin et
       - EKF'ye LightGlue tahmini olarak besle

Mimari:
    frame → SuperPoint → LightGlue → koordinat tahmini
                                  ↓
                            referans harita
"""

import sys
from pathlib import Path
import os
import cv2
import torch
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# LightGlue kütüphanesini yerel klasörden yüklemek için sys.path'e ekle
try:
    _lg_path = os.path.join(os.path.dirname(__file__), "LightGlue")
    if _lg_path not in sys.path:
        sys.path.insert(0, _lg_path)
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    LIGHTGLUE_AVAILABLE = True
    logger.info("LightGlue kurulu ✅")
except ImportError as e:
    LIGHTGLUE_AVAILABLE = False
    logger.warning(f"LightGlue kurulu değil: {e}")


# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ReferenceFrame:
    """Haritadaki tek bir referans nokta."""
    frame_id:    int
    gps_x:       float
    gps_y:       float
    gps_z:       float
    keypoints:   np.ndarray        # (N, 2) — piksel koordinatları
    descriptors: np.ndarray        # (N, D) — feature vektörleri
    thumbnail:   Optional[np.ndarray] = None  # debug için küçük görüntü


@dataclass 
class MatchResult:
    """Tek bir eşleştirme sonucu."""
    matched_frame_id: int
    estimated_x:      float
    estimated_y:      float
    estimated_z:      float
    confidence:       float        # 0-1 arası, eşleşme kalitesi
    num_matches:      int
    inlier_ratio:     float


# ─────────────────────────────────────────────────────────────────────────── #

class LightGlueMatcher:
    """
    LightGlue tabanlı görüntü eşleştirme motoru.

    frame_processor.py → visual_odometry.py → LightGlueMatcher
    """

    MIN_KEYPOINTS  = 50     # En az bu kadar keypoint olmalı
    MIN_MATCHES    = 20     # En az bu kadar eşleşme olmalı
    MAP_INTERVAL   = 5      # Her N frame'de bir haritaya ekle (GPS sağlıklı)
    TOP_K_MATCHES  = 5      # En iyi K eşleşmeyi kullan
    MAX_MAP_SIZE   = 200    # Haritada max bu kadar frame tut

    def __init__(self, cfg: dict):
        """
        cfg: config.yaml task2.lightglue blogu

        Örnek:
            cfg = {
                "device": "cuda",
                "max_keypoints": 1024,
                "map_interval": 5,
                "min_matches": 20,
            }
        """
        self.cfg = cfg
        self._map_interval  = cfg.get("map_interval", self.MAP_INTERVAL)
        self._min_matches   = cfg.get("min_matches",  self.MIN_MATCHES)
        self._max_map_size  = cfg.get("max_map_size", self.MAX_MAP_SIZE)
        self._max_keypoints = cfg.get("max_keypoints", 1024)

        # Referans harita: frame_id → ReferenceFrame
        self._reference_map: Dict[int, ReferenceFrame] = {}
        self._frame_count = 0

        # Model ve cihaz
        self._device:     Any = None
        self._extractor:  Any = None
        self._matcher:    Any = None

        # Fallback: LightGlue yoksa ORB kullan
        self._use_orb_fallback = False
        self._orb: Any = None
        self._bf:  Any = None

        self._initialize_models()

    # ────────────────────────────────────────────────────────────── #

    def _initialize_models(self):
        """Model ve cihazı başlat."""
        if LIGHTGLUE_AVAILABLE:
            try:
                import torch
                device_str = self.cfg.get("device", "cuda")
                if device_str == "cuda" and not torch.cuda.is_available():
                    device_str = "cpu"
                    logger.warning("CUDA yok, CPU kullanılıyor")

                self._device = torch.device(device_str)

                import lightglue as _lg  # type: ignore[import]
                self._extractor = _lg.SuperPoint(max_num_keypoints=self._max_keypoints).eval().to(self._device)
                self._matcher   = _lg.LightGlue(features="superpoint").eval().to(self._device)

                logger.info(f"LightGlue başlatıldı — device: {self._device}")

            except Exception as e:
                logger.error(f"LightGlue başlatma hatası: {e}")
                self._init_orb_fallback()
        else:
            self._init_orb_fallback()

    def _init_orb_fallback(self):
        """LightGlue yoksa ORB + BFMatcher kullan."""
        self._use_orb_fallback = True
        self._orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        logger.warning("ORB fallback aktif (LightGlue önerilir)")

    # ────────────────────────────────────────────────────────────── #

    def add_to_map(self, frame: np.ndarray,
                          gps_x: float, gps_y: float, gps_z: float) -> bool:
        """
        GPS sağlıklı fazda çağrılır.
        Frame'i feature'larıyla birlikte referans haritaya ekle.

        Returns: True → başarıyla eklendi, False → yeterli feature yok
        """
        self._frame_count += 1

        # Her MAP_INTERVAL frame'de bir ekle (hafıza tasarrufu)
        if self._frame_count % self._map_interval != 0:
            return False

        # Feature çıkar
        kpts, descs = self._extract_features(frame)

        if kpts is None or descs is None or len(kpts) < self.MIN_KEYPOINTS:
            logger.debug(f"Frame {self._frame_count}: yeterli keypoint yok ({len(kpts) if kpts is not None else 0})")
            return False

        # Harita boyutu kontrolü — en eski frame'i sil
        if len(self._reference_map) >= self._max_map_size:
            oldest_key = min(self._reference_map.keys())
            del self._reference_map[oldest_key]

        # Küçük thumbnail oluştur (debug için)
        thumbnail = cv2.resize(frame, (160, 90)) if frame is not None else None

        ref = ReferenceFrame(
            frame_id    = self._frame_count,
            gps_x       = gps_x,
            gps_y       = gps_y,
            gps_z       = gps_z,
            keypoints   = kpts,
            descriptors = descs,
            thumbnail   = thumbnail,
        )

        self._reference_map[self._frame_count] = ref

        logger.debug(
            f"Haritaya eklendi: frame={self._frame_count} | "
            f"GPS=({gps_x:.2f}, {gps_y:.2f}, {gps_z:.2f}) | "
            f"Harita boyutu: {len(self._reference_map)}"
        )
        return True

    # ────────────────────────────────────────────────────────────── #

    def estimate_position(self, frame: np.ndarray,
                          est_gps_x: float = 0.0,
                          est_gps_y: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
        """
        GPS kesilince çağrılır.
        Gelen frame'i referans haritayla eşleştir, koordinat tahmin et.

        Returns: (x, y, z, confidence) veya None
        """
        if len(self._reference_map) < 3:
            logger.warning("Harita çok küçük, eşleştirme yapılamıyor")
            return None

        # Sorgu frame'inden feature çıkar
        query_kpts, query_descs = self._extract_features(frame)

        if query_kpts is None or query_descs is None or len(query_kpts) < self.MIN_KEYPOINTS:
            return None

        # Tüm haritayı taramak CPU'yu kilitler. Global lokalizasyon için
        # harita boyunca eşit dağılmış en fazla 10 frame seçiyoruz.
        match_results: List[MatchResult] = []
        all_ids = list(self._reference_map.keys())
        stride = max(1, len(all_ids) // 10)
        candidate_ids = all_ids[::stride][:10]

        for frame_id in candidate_ids:
            ref_frame = self._reference_map[frame_id]
            result = self._match_with_reference(
                query_kpts, query_descs,
                ref_frame
            )
            if result is not None:
                match_results.append(result)

        if not match_results:
            logger.warning("Hiçbir referans frame ile eşleşme bulunamadı")
            return None

        # En iyi K eşleşmeyi seç
        match_results.sort(key=lambda r: r.confidence, reverse=True)
        top_matches = match_results[:self.TOP_K_MATCHES]

        # Ağırlıklı ortalama koordinat
        x, y, z, total_conf = self._weighted_position(top_matches)

        logger.info(
            f"LightGlue tahmin: ({x:.2f}, {y:.2f}, {z:.2f}) | "
            f"conf={total_conf:.3f} | {len(top_matches)} eşleşme"
        )

        return x, y, z, total_conf

    # ────────────────────────────────────────────────────────────── #

    def _extract_features(self, frame: np.ndarray):
        """Frame'den keypoint ve descriptor çıkar."""
        if frame is None:
            return None, None

        # Gri tonlamaya çevir
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self._use_orb_fallback:
            return self._extract_orb(gray)
        else:
            return self._extract_superpoint(gray)

    def _extract_superpoint(self, gray: np.ndarray):
        """SuperPoint ile feature çıkar."""
        try:
            import torch

            # Normalize et ve tensor'a çevir
            img_tensor = torch.from_numpy(gray).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self._device)

            with torch.no_grad():
                feats = self._extractor.extract(img_tensor)

            kpts  = feats["keypoints"][0].cpu().numpy()   # (N, 2)
            descs = feats["descriptors"][0].cpu().numpy()  # (N, D)

            return kpts, descs

        except Exception as e:
            logger.error(f"SuperPoint hatası: {e}")
            return self._extract_orb(gray)

    def _extract_orb(self, gray: np.ndarray):
        """ORB ile feature çıkar (fallback)."""
        try:
            kpts_cv, descs = self._orb.detectAndCompute(gray, None)
            if kpts_cv is None or len(kpts_cv) == 0:
                return None, None
            kpts = np.array([[k.pt[0], k.pt[1]] for k in kpts_cv])
            return kpts, descs
        except Exception as e:
            logger.error(f"ORB hatası: {e}")
            return None, None

    # ────────────────────────────────────────────────────────────── #

    def _match_with_reference(self,
                               query_kpts:  np.ndarray,
                               query_descs: np.ndarray,
                               ref_frame:   ReferenceFrame) -> Optional[MatchResult]:
        """Sorgu frame'ini tek bir referans frame ile eşleştir."""
        try:
            if self._use_orb_fallback:
                return self._match_orb(query_kpts, query_descs, ref_frame)
            else:
                return self._match_lightglue(query_kpts, query_descs, ref_frame)
        except Exception as e:
            logger.debug(f"Eşleştirme hatası frame {ref_frame.frame_id}: {e}")
            return None

    def _match_lightglue(self,
                          query_kpts:  np.ndarray,
                          query_descs: np.ndarray,
                          ref_frame:   ReferenceFrame) -> Optional[MatchResult]:
        """LightGlue ile eşleştir."""
        import torch

        def _to_tensor(kpts, descs):
            return {
                "keypoints":   torch.from_numpy(kpts).float().unsqueeze(0).to(self._device),
                "descriptors": torch.from_numpy(descs).float().unsqueeze(0).to(self._device),
            }

        feats0 = _to_tensor(query_kpts,       query_descs)
        feats1 = _to_tensor(ref_frame.keypoints, ref_frame.descriptors)

        with torch.no_grad():
            matches_out = self._matcher({"image0": feats0, "image1": feats1})

        matches   = matches_out["matches"][0].cpu().numpy()     # (M, 2)
        scores    = matches_out["matching_scores0"][0].cpu().numpy()  # (N,)

        num_matches = len(matches)
        if num_matches < self._min_matches:
            return None

        # İnlier oranı (yüksek score'lu eşleşmeler)
        high_score = np.sum(scores > 0.5)
        inlier_ratio = high_score / max(len(scores), 1)

        # Confidence = eşleşme sayısı + inlier oranı
        confidence = min(1.0, (num_matches / 100.0) * inlier_ratio)

        return MatchResult(
            matched_frame_id = ref_frame.frame_id,
            estimated_x      = ref_frame.gps_x,
            estimated_y      = ref_frame.gps_y,
            estimated_z      = ref_frame.gps_z,
            confidence       = confidence,
            num_matches      = num_matches,
            inlier_ratio     = inlier_ratio,
        )

    def _match_orb(self,
                    query_kpts:  np.ndarray,
                    query_descs: np.ndarray,
                    ref_frame:   ReferenceFrame) -> Optional[MatchResult]:
        """ORB + BFMatcher ile eşleştir (fallback)."""
        if ref_frame.descriptors is None:
            return None

        matches = self._bf.knnMatch(query_descs, ref_frame.descriptors, k=2)

        # Lowe's ratio test
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)
        if num_matches < self._min_matches:
            return None

        inlier_ratio = num_matches / max(len(matches), 1)
        confidence   = min(1.0, num_matches / 80.0 * inlier_ratio)

        return MatchResult(
            matched_frame_id = ref_frame.frame_id,
            estimated_x      = ref_frame.gps_x,
            estimated_y      = ref_frame.gps_y,
            estimated_z      = ref_frame.gps_z,
            confidence       = confidence,
            num_matches      = num_matches,
            inlier_ratio     = inlier_ratio,
        )

    # ────────────────────────────────────────────────────────────── #

    def _weighted_position(self,
                            matches: List[MatchResult]
                            ) -> Tuple[float, float, float, float]:
        """
        En iyi eşleşmelerden ağırlıklı ortalama koordinat hesapla.
        Confidence değerleri ağırlık olarak kullanılır.
        """
        total_conf = sum(m.confidence for m in matches)

        if total_conf < 1e-6:
            # Tüm confidence sıfırsa basit ortalama al
            x = np.mean([m.estimated_x for m in matches])
            y = np.mean([m.estimated_y for m in matches])
            z = np.mean([m.estimated_z for m in matches])
            return float(x), float(y), float(z), 0.0

        x = sum(m.estimated_x * m.confidence for m in matches) / total_conf
        y = sum(m.estimated_y * m.confidence for m in matches) / total_conf
        z = sum(m.estimated_z * m.confidence for m in matches) / total_conf

        # Normalize confidence (0-1)
        avg_conf = total_conf / len(matches)

        return float(x), float(y), float(z), float(min(avg_conf, 1.0))

    # ────────────────────────────────────────────────────────────── #

    @property
    def map_size(self) -> int:
        """Haritadaki referans frame sayısı."""
        return len(self._reference_map)

    @property
    def is_ready(self) -> bool:
        """Eşleştirme için yeterli harita var mı?"""
        return len(self._reference_map) >= 3

    def get_map_stats(self) -> dict:
        """Harita istatistikleri (debug için)."""
        if not self._reference_map:
            return {"size": 0}

        gps_coords = [(r.gps_x, r.gps_y) for r in self._reference_map.values()]
        xs = [c[0] for c in gps_coords]
        ys = [c[1] for c in gps_coords]

        return {
            "size":      len(self._reference_map),
            "x_range":   (min(xs), max(xs)),
            "y_range":   (min(ys), max(ys)),
            "frame_ids": list(self._reference_map.keys())[-5:],  # son 5
        }

    def reset(self):
        """Haritayı ve sayaçları sıfırla (yeni oturum)."""
        self._reference_map.clear()
        self._frame_count = 0
        logger.info("LightGlueMatcher sıfırlandı")