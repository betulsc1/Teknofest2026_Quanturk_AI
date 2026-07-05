"""
training/convert_teknofest_to_yolo.py

TEKNOFEST Pascal VOC XML etiketlerini YOLO formatına çevirir.

Kullanım:
    python training/convert_teknofest_to_yolo.py \
        --input_dir "D:/TEKNOFEST HYZ 2025 Verileri/THYZ_2025_Oturum_1" \
        --output_dir "D:/solid/archive/TEKNOFEST2025/Oturum_1"

Klasör yapısı beklentisi:
    THYZ_2025_Oturum_1/
        Images/          ← PNG görüntüler
        Annotations/     ← XML etiketler

Çıktı:
    output_dir/
        images/          ← PNG görüntüler (kopyalanır)
        labels/          ← YOLO .txt etiketler
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

# TEKNOFEST sınıf → YOLO class ID mapping
# Türkçe karakter varyasyonlarını da ekledik
CLASS_MAP = {
    # Taşıt varyasyonları
    "taşıt": 0,
    "tasit": 0,
    "Taşıt": 0,
    "Tasit": 0,
    "vehicle": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,

    # İnsan varyasyonları
    "i̇nsan": 1,
    "insan": 1,
    "İnsan": 1,
    "Insan": 1,
    "person": 1,
    "pedestrian": 1,
    "human": 1,

    # UAP varyasyonları
    "uap": 2,
    "UAP": 2,
    "Uap": 2,
    "uçan araba park": 2,
    "park alani": 2,

    # UAİ varyasyonları
    "uai": 3,
    "UAİ": 3,
    "UAI": 3,
    "Uai": 3,
    "uçan ambulans iniş": 3,
    "inis alani": 3,
}


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list:
    """
    Tek bir XML dosyasını YOLO formatına çevirir.
    Çıktı: [(class_id, cx, cy, w, h), ...]  normalize edilmiş
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  XML parse hatasi: {xml_path.name} → {e}")
        return []

    # Boyutları XML'den de alabiliriz ama parametre öncelikli
    size_elem = root.find("size")
    if size_elem is not None:
        w_el = size_elem.find("width")
        h_el = size_elem.find("height")
        if w_el is not None and w_el.text:
            img_w = int(float(w_el.text))
        if h_el is not None and h_el.text:
            img_h = int(float(h_el.text))

    annotations = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or name_elem.text is None:
            continue

        raw_name = name_elem.text.strip()

        # Sınıf ID bul
        class_id = CLASS_MAP.get(raw_name)
        if class_id is None:
            # Küçük harf deneme
            class_id = CLASS_MAP.get(raw_name.lower())
        if class_id is None:
            print(f"  Bilinmeyen sinif: '{raw_name}' → atlanıyor")
            continue

        # Bounding box
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin_el = bndbox.find("xmin")
            ymin_el = bndbox.find("ymin")
            xmax_el = bndbox.find("xmax")
            ymax_el = bndbox.find("ymax")
            if any(el is None for el in [xmin_el, ymin_el, xmax_el, ymax_el]):
                continue
            xmin = float(xmin_el.text or 0)  # type: ignore[union-attr]
            ymin = float(ymin_el.text or 0)  # type: ignore[union-attr]
            xmax = float(xmax_el.text or 0)  # type: ignore[union-attr]
            ymax = float(ymax_el.text or 0)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            continue

        # Sınır kontrolü
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        # YOLO normalize (cx, cy, w, h)
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        annotations.append((class_id, cx, cy, bw, bh))

    return annotations


def convert_session(input_dir: str, output_dir: str, session_name: str = ""):
    """
    Tek bir oturum klasörünü dönüştürür.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    # Görüntü ve etiket klasörleri
    img_out = output_path / "images"
    lbl_out = output_path / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Görüntü klasörü bul (Images veya frames veya direkt klasör)
    img_dir = None
    for candidate in ["Images", "images", "frames", "Frames", "JPEGImages"]:
        if (input_path / candidate).exists():
            img_dir = input_path / candidate
            break
    if img_dir is None:
        # Görüntüler direkt klasörde mi?
        pngs = list(input_path.glob("*.PNG")) + list(input_path.glob("*.jpg"))
        if pngs:
            img_dir = input_path
        else:
            print(f"Goruntu klasoru bulunamadi: {input_path}")
            return 0, 0

    # Etiket klasörü bul
    ann_dir = None
    for candidate in ["Annotations", "annotations", "labels", "Labels"]:
        if (input_path / candidate).exists():
            ann_dir = input_path / candidate
            break
    if ann_dir is None:
        print(f"Etiket klasoru bulunamadi: {input_path}")
        return 0, 0

    # Görüntüleri işle
    img_files = (list(img_dir.glob("*.PNG")) + list(img_dir.glob("*.png")) +
                 list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")))

    converted = 0
    skipped   = 0
    total_obj = 0

    print(f"\n{'='*50}")
    print(f"Oturum : {session_name or input_path.name}")
    print(f"Goruntu: {len(img_files)}")
    print(f"{'='*50}")

    for img_file in sorted(img_files):
        stem = img_file.stem  # frame_000888

        # XML etiket dosyasını bul
        xml_file = ann_dir / f"{stem}.xml"
        if not xml_file.exists():
            # Küçük harf dene
            xml_file = ann_dir / f"{stem.lower()}.xml"
        if not xml_file.exists():
            skipped += 1
            continue

        # Dönüştür
        yolo_lines = xml_to_yolo(xml_file, img_w=3840, img_h=2160)

        # Görüntüyü kopyala
        out_img = img_out / img_file.name
        if not out_img.exists():
            shutil.copy2(img_file, out_img)

        # YOLO label yaz (boş da olsa yaz — background frame)
        lbl_file = lbl_out / f"{stem}.txt"
        with open(lbl_file, "w", encoding="utf-8") as f:
            for cls_id, cx, cy, bw, bh in yolo_lines:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1
        total_obj += len(yolo_lines)

    print(f"Donusturulen : {converted} frame")
    print(f"Atlanan      : {skipped} frame (XML yok)")
    print(f"Toplam nesne : {total_obj}")

    return converted, total_obj


def main():
    parser = argparse.ArgumentParser(description="TEKNOFEST VOC → YOLO Donusturucu")
    parser.add_argument("--input_dir",  required=True, help="Oturum klasoru (THYZ_2025_Oturum_X)")
    parser.add_argument("--output_dir", required=True, help="Cikti klasoru")
    parser.add_argument("--session",    default="",    help="Oturum adi (log icin)")
    args = parser.parse_args()

    converted, total_obj = convert_session(
        args.input_dir, args.output_dir, args.session
    )

    print(f"\nTamamlandi! {converted} frame, {total_obj} nesne")


if __name__ == "__main__":
    main()