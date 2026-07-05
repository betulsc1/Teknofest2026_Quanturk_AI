"""
src/task2_position/optical_flow.py

RAFT tabanli optik akis hesaplama.
GPU yoksa otomatik olarak Farneback (CPU) kullanir.
"""

import cv2  # type: ignore[import]
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore[import]
    import torchvision  # type: ignore[import]
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OpticalFlowEstimator:
    """
    Optik akis motoru.
    RAFT (GPU) varsa RAFT, yoksa Farneback (CPU) kullanir.
    """

    def __init__(self, cfg: dict):
        self.raft_iters = int(cfg.get("raft_iters", 12))
        self.device     = str(cfg.get("device", "cpu"))
        self.use_raft   = bool(cfg.get("use_raft", True)) and TORCH_AVAILABLE

        # Frame küçültme oranı (1.0 = orijinal, 0.5 = yarı boyut)
        # Küçültme gürültüyü azaltır ve hızlandırır
        self.downscale  = float(cfg.get("flow_downscale", 0.5))

        self._raft_model  = None
        self._transforms  = None

        if self.use_raft:
            self._load_raft()

    # ------------------------------------------------------------------ #

    def _load_raft(self) -> None:
        try:
            from torchvision.models.optical_flow import (  # type: ignore[import]
                raft_large, Raft_Large_Weights
            )
            weights          = Raft_Large_Weights.DEFAULT
            self._raft_model = raft_large(weights=weights, progress=False)
            self._raft_model = self._raft_model.to(self.device)  # type: ignore[union-attr]
            self._raft_model.eval()  # type: ignore[union-attr]
            self._transforms = weights.transforms()
            logger.info(f"RAFT yuklendi | Cihaz: {self.device}")
        except Exception as e:
            logger.warning(f"RAFT yuklenemedi ({e}), Farneback kullanilacak")
            self.use_raft    = False
            self._raft_model = None

    # ------------------------------------------------------------------ #

    def compute(self, prev_frame: np.ndarray,
                curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Iki frame arasindaki dense optical flow hesaplar.
        Cikti: (H, W, 2) numpy array ya da None.
        """
        if prev_frame is None or curr_frame is None:
            return None

        if self.use_raft and self._raft_model is not None:
            return self._compute_raft(prev_frame, curr_frame)
        return self._compute_farneback(prev_frame, curr_frame)

    # ------------------------------------------------------------------ #

    def _compute_raft(self, prev_frame: np.ndarray,
                      curr_frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
            curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]

            prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).float()  # type: ignore[possibly-undefined]
            curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).float()  # type: ignore[possibly-undefined]

            prev_t, curr_t = self._transforms(prev_t, curr_t)  # type: ignore[misc]
            prev_t = prev_t.unsqueeze(0).to(self.device)
            curr_t = curr_t.unsqueeze(0).to(self.device)

            with torch.no_grad():  # type: ignore[possibly-undefined]
                preds = self._raft_model(prev_t, curr_t)  # type: ignore[operator]

            flow: np.ndarray = preds[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            return flow
        except Exception as e:
            logger.warning(f"RAFT hatasi: {e}")
            return self._compute_farneback(prev_frame, curr_frame)

    # ------------------------------------------------------------------ #

    def _compute_farneback(self, prev_frame: np.ndarray,
                           curr_frame: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]

        # Frame küçültme — gürültüyü azaltır, akışı temizler
        if self.downscale < 1.0:
            h, w = prev_gray.shape
            new_w = int(w * self.downscale)
            new_h = int(h * self.downscale)
            prev_small = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            prev_small = prev_gray
            curr_small = curr_gray

        flow_out = np.zeros((*prev_small.shape, 2), dtype=np.float32)
        flow: np.ndarray = cv2.calcOpticalFlowFarneback(  # type: ignore[attr-defined]
            prev_small, curr_small,
            flow_out,
            pyr_scale=0.5,
            levels=5,         # Daha fazla piramit seviyesi (3→5)
            winsize=21,       # Daha geniş pencere (15→21)
            iterations=5,     # Daha fazla iterasyon (3→5)
            poly_n=7,         # Daha geniş polinom (5→7)
            poly_sigma=1.5,   # Daha yumuşak (1.2→1.5)
            flags=0,
        )

        # Piksel kaymasını orijinal çözünürlüğe geri ölçekle
        if self.downscale < 1.0:
            flow = flow / self.downscale
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)

        return flow

    # ------------------------------------------------------------------ #

    def get_background_flow(self, flow: np.ndarray,
                            detection_masks: Optional[np.ndarray] = None
                            ) -> Tuple[float, float, float]:
        """Arkaplan akisini hesapla. Donus: (dx, dy, confidence)"""
        if flow is None:
            return 0.0, 0.0, 0.0

        h, w = flow.shape[:2]
        mask: np.ndarray = (detection_masks == 0) if detection_masks is not None \
                           else np.ones((h, w), dtype=bool)

        magnitude: np.ndarray = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        valid_mask: np.ndarray = mask & (magnitude < 100)
        valid_count = int(valid_mask.sum())
        confidence  = valid_count / (h * w)

        if valid_count < 100:
            return 0.0, 0.0, 0.0

        dx = float(np.median(flow[valid_mask, 0]))
        dy = float(np.median(flow[valid_mask, 1]))
        return dx, dy, confidence

    # ------------------------------------------------------------------ #

    def get_flow_quality(self, flow: np.ndarray) -> float:
        """Akis kalitesi: 0-1."""
        if flow is None:
            return 0.0
        magnitude: np.ndarray = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_mag = float(magnitude.mean())
        std_mag  = float(magnitude.std())
        if mean_mag < 0.1:
            return 0.3
        if std_mag / (mean_mag + 1e-6) > 3.0:
            return 0.2
        return float(min(1.0, 1.0 / (1.0 + std_mag / mean_mag)))