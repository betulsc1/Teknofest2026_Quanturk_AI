"""
scripts/run_competition.py

TEKNOFEST 2026 Havacılıkta Yapay Zeka — Yarışma Başlat Scripti

Yarışma günü tek komutla tüm sistemi çalıştırır:
    python scripts/run_competition.py

Yarışma günü doldurulacak dosyalar:
    - config/server_config.yaml   (sunucu IP ve token)
    - config/camera_params.yaml   (yarışma gününde verilecek fx, fy, cx, cy)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

# ── Proje kökünü path'e ekle ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Communication ─────────────────────────────────────────────────────
from src.communication.api_client    import CompetitionAPIClient
from src.communication.frame_fetcher import FrameFetcher
from src.communication.result_sender import ResultSender

# ── Core ──────────────────────────────────────────────────────────────
from src.core.frame_processor import FrameProcessor
from src.core.session_manager import SessionManager

# ── Görev 1 ───────────────────────────────────────────────────────────
from src.task1_detection.detector         import ObjectDetector
from src.task1_detection.motion_detector  import MotionDetector
from src.task1_detection.landing_checker  import LandingChecker

# ── Görev 2 ───────────────────────────────────────────────────────────
from src.task2_position.visual_odometry   import VisualOdometry

# ── Görev 3 ───────────────────────────────────────────────────────────
from src.task3_matching.matcher           import ReferenceMatcher


# ═══════════════════════════════════════════════════════════════════════ #
#  Loglama + CLI
# ═══════════════════════════════════════════════════════════════════════ #

def setup_logging(debug: bool = False):
    log_level = logging.DEBUG if debug else logging.INFO
    log_dir   = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"competition_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Log dosyası: {log_file}")


def parse_args():
    p = argparse.ArgumentParser(
        description="TEKNOFEST 2026 Havacılıkta Yapay Zeka — Yarışma Sistemi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server-config", default="config/server_config.yaml")
    p.add_argument("--camera-config", default="config/camera_params.yaml")
    p.add_argument("--task-config",   default="config/config.yaml",
                   help="Görev parametreleri (task1, task2, task3 blokları)")
    p.add_argument("--model",         default="models/task1/detector/best.pt")

    p.add_argument("--conf",  type=float, default=0.35, help="YOLO confidence")
    p.add_argument("--iou",   type=float, default=0.45, help="NMS IoU")
    p.add_argument("--imgsz", type=int,   default=1280, help="YOLO image size")

    p.add_argument("--no-sahi", action="store_true", help="SAHI kapalı")
    p.add_argument("--device",  default="auto",
                   choices=["auto", "cuda:0", "cuda:1", "cpu"])

    p.add_argument("--mode", default="competition",
                   choices=["competition", "test", "dry-run"])
    p.add_argument("--debug", action="store_true",
                   help="Debug modu (görselleştirme + detaylı log)")

    p.add_argument("--no-cls-url", action="store_true",
                   help="cls alanını URL yerine düz string olarak gönder")

    return p.parse_args()


def load_yaml(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        logging.warning(f"Config bulunamadı: {path}, boş dict kullanılıyor")
        return {}
    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════ #
#  Setup doğrulama
# ═══════════════════════════════════════════════════════════════════════ #

def validate_setup(args, server_cfg: dict) -> bool:
    errors, warnings = [], []

    model_path = Path(args.model)
    if not model_path.exists():
        errors.append(f"Model dosyası bulunamadı: {model_path}")

    srv = server_cfg.get("server", {})
    token = srv.get("token", "")
    if not token or token in ("TAKIM_TOKEN", "YOUR_TEAM_TOKEN_HERE"):
        warnings.append("server_config.yaml: Takım token doldurulmamış!")

    url = srv.get("url", "")
    if not url or "127.0.0.25" in url or "127.0.0.1" in url:
        warnings.append(f"server_config.yaml: Sunucu URL'i ({url}) kontrol edin!")

    if warnings:
        print("\n⚠️  Uyarılar:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n❌ Hatalar:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\n✅ Setup OK\n")
    return True


# ═══════════════════════════════════════════════════════════════════════ #
#  Yarışma döngüsü
# ═══════════════════════════════════════════════════════════════════════ #

def run_competition(args):
    logger = logging.getLogger("competition")

    # 1. Config yükle
    server_cfg = load_yaml(args.server_config)
    camera_cfg = load_yaml(args.camera_config).get("camera", {})
    task_cfg   = load_yaml(args.task_config)
    task1_cfg  = task_cfg.get("task1", {})
    task2_cfg  = task_cfg.get("task2", {})
    task3_cfg  = task_cfg.get("task3", {})

    if args.mode == "competition" and not validate_setup(args, server_cfg):
        sys.exit(1)

    srv        = server_cfg.get("server", {})
    server_url = srv.get("url", "http://127.0.0.1:5000")
    token      = srv.get("token", "")
    timeout    = srv.get("timeout", 5)
    device     = resolve_device(args.device)

    logger.info(f"Sunucu: {server_url} | Cihaz: {device}")

    # 2. ApiClient + bağlantı testi
    api = CompetitionAPIClient(
        server_url=server_url,
        token=token,
        timeout=timeout,
        cls_as_url=not args.no_cls_url,
    )

    if args.mode in ("competition", "test"):
        logger.info("Sunucu bağlantısı test ediliyor...")
        if not api.test_connection():
            logger.error("Bağlantı başarısız, çıkılıyor.")
            sys.exit(2)
        logger.info("Bağlantı OK")

    # 3. Görev 1 modülleri
    logger.info("Görev 1 modülleri yükleniyor...")
    detector = ObjectDetector({
        "model_path":    args.model,
        "confidence":    args.conf,
        "iou_threshold": args.iou,
        "image_size":    args.imgsz,
        "device":        device,
    })
    if not detector.is_loaded:
        logger.error("Detector yüklenemedi!")
        sys.exit(3)

    if not args.no_sahi:
        detector.enable_sahi(
            slice_size=task1_cfg.get("sahi_slice", 640),
            overlap=task1_cfg.get("sahi_overlap", 0.2),
        )
    detector.warmup()

    motion_det  = MotionDetector({
        "threshold_px":  task1_cfg.get("motion_threshold_px", 8),
        "min_bg_points": task1_cfg.get("min_bg_points", 50),
    })
    landing_chk = LandingChecker(
        edge_margin=task1_cfg.get("landing_edge_margin", 5),
        overlap_ratio=task1_cfg.get("landing_overlap_ratio", 0.05),
    )

    # 4. Görev 2 (VisualOdometry)
    logger.info("Görev 2 modülleri yükleniyor...")
    vo = VisualOdometry(task2_cfg=task2_cfg, camera_cfg=camera_cfg)

    # 5. Görev 3 (Matcher)
    logger.info("Görev 3 modülleri yükleniyor...")
    matcher = ReferenceMatcher(task3_cfg)

    # 6. Communication yardımcıları
    fetcher = FrameFetcher(api_client=api, buffer_size=5)
    sender  = ResultSender(api_client=api, max_retries=3, retry_delay=0.5)

    # 7. FrameProcessor (DI)
    processor = FrameProcessor(
        detector=detector,
        motion_detector=motion_det,
        landing_checker=landing_chk,
        visual_odometry=vo,
        matcher=matcher,
        debug=args.debug,
    )

    # 8. Session + referans nesneler
    session = SessionManager()

    if args.mode == "dry-run":
        logger.info("✅ Dry-run başarılı — sistem hazır")
        return

    # Referans nesneleri (Görev 3) yükle
    refs = api.get_reference_objects()
    if refs:
        matcher.load_references(refs)

    # Frame listesi çek
    frames = api.get_frame_list()
    if not frames:
        logger.error("Frame listesi boş, çıkılıyor.")
        sys.exit(4)
    session.load_frames(frames)

    total = session.total_frames
    logger.info(f"═══ Yarışma başlıyor: {total} frame ═══")

    # 9. Ana döngü
    start = time.perf_counter()
    debug_writer = None
    out_path = ROOT / "logs" / f"debug_{datetime.now():%Y%m%d_%H%M%S}.mp4"

    while True:
        frame_data = session.next_frame()
        if frame_data is None:
            break

        idx = session.current_idx - 1
        session.update_gps_status(frame_data)

        try:
            # Görüntü indir
            fetched = fetcher.fetch(frame_data)
            frame   = fetched["frame"]
            prev    = fetcher.previous_frame()

            # İşle
            out = processor.process(
                frame=frame,
                prev_frame=prev,
                frame_data=frame_data,
                frame_idx=idx,
                total_frames=total,
            )

            # Gönder
            ok = sender.send(frame_data["url"], out["result"])
            if ok:
                session.mark_sent(frame_data["url"])

            # Debug görselleştirme
            if args.debug and out.get("debug_frame") is not None:
                if debug_writer is None:
                    h, w = out["debug_frame"].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                    debug_writer = cv2.VideoWriter(
                        str(out_path), fourcc, 7.5, (w, h)
                    )
                debug_writer.write(out["debug_frame"])

            # İlerleme
            if idx % 50 == 0 or idx == total - 1:
                elapsed = time.perf_counter() - start
                fps     = (idx + 1) / max(elapsed, 1e-6)
                remain  = (total - idx - 1) / max(fps, 1e-6)
                logger.info(
                    f"[{idx+1:4d}/{total}] "
                    f"FPS: {fps:.2f} | "
                    f"Frame: {out['elapsed_ms']:.0f}ms | "
                    f"Det: {out['det_count']} | "
                    f"GPS: {'OK' if out['gps_healthy'] else 'KESIK'} | "
                    f"Kalan: {remain:.0f}s"
                )

        except Exception as e:
            logger.error(f"Frame #{idx} hata: {e}", exc_info=args.debug)
            # Boş sonuç gönder, frame atlanmasın
            empty = {
                "detections": [],
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "matched_objects": [],
            }
            sender.send(frame_data["url"], empty)

    # 10. Bitiş raporu
    total_time = time.perf_counter() - start
    stats = sender.stats()
    vo_stats = vo.get_status()

    if debug_writer:
        debug_writer.release()

    logger.info("═" * 60)
    logger.info("YARIŞMA TAMAMLANDI")
    logger.info(f"Süre:          {total_time:.1f}s ({total_time/60:.1f} dk)")
    logger.info(f"Toplam frame:  {total}")
    logger.info(f"Ortalama FPS:  {total/max(total_time, 1e-6):.2f}")
    logger.info(f"Gönderilen:    {stats['sent']}")
    logger.info(f"Başarısız:     {stats['failed']}")
    logger.info(f"Ort. istek:    {stats['avg_ms']:.2f} ms")
    logger.info(f"VO durum:      {vo_stats}")
    logger.info(f"Oturum:        {session.summary()}")
    logger.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════════ #

def print_banner():
    print(r"""
    ╔════════════════════════════════════════════════════════╗
    ║  TEKNOFEST 2026 — Havacılıkta Yapay Zeka               ║
    ║  Takım: QUANTÜRK                                       ║
    ╚════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()
    args = parse_args()
    setup_logging(debug=args.debug)

    try:
        run_competition(args)
    except KeyboardInterrupt:
        logging.warning("Kullanıcı durdurdu (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Kritik hata: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()