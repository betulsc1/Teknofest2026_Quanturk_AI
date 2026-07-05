"""
src/utils/logger.py
Projenin tek merkezi loglama sistemi.
Her modül buradan logger alır, print() kullanmaz.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "teknofest",
                 level: int = logging.INFO,
                 log_to_file: bool = True) -> logging.Logger:
    """
    Proje genelinde kullanılacak logger'ı oluşturur ve döndürür.

    Kullanım (herhangi bir modülde):
        from src.utils.logger import setup_logger
        logger = setup_logger(__name__)
        logger.info("Mesaj")
        logger.warning("Uyarı")
        logger.error("Hata")
    """
    logger = logging.getLogger(name)

    # Aynı logger tekrar kurulmasın
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # --- Konsol çıktısı ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Dosya çıktısı (yarışmada hata ayıklamak için kritik) ---
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{timestamp}_{name}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)   # Dosyaya her şeyi yaz
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Modül düzeyinde varsayılan logger — doğrudan import edilebilir
log = setup_logger("teknofest")