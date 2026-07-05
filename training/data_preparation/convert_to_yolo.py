"""
Pascal VOC XML → YOLO format dönüştürücü
THYZ_2025 veri seti için

Kullanım:
    python training/data_preparation/convert_to_yolo.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import yaml
from tqdm import tqdm

# Sınıf mapping — XML'deki Türkçe isimler
CLASS_MAP = {
    'Taşıt': 0, 'tasit': 0, 'Tasit': 0,
    'İnsan': 1, 'insan': 1, 'Insan': 1,
    'UAP': 2, 'uap': 2,
    'UAİ': 3, 'UAI': 3, 'uai': 3,
}

def convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax):
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h

def convert_xml_to_yolo(xml_path: Path, out_txt: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is not None:
        w_el = size.find('width')
        h_el = size.find('height')
        img_w = int(float(w_el.text)) if w_el is not None and w_el.text else 3840
        img_h = int(float(h_el.text)) if h_el is not None and h_el.text else 2160
    else:
        img_w, img_h = 3840, 2160

    lines = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None or name_el.text is None:
            continue
        name = name_el.text.strip()
        cls_id = CLASS_MAP.get(name)
        if cls_id is None:
            print(f"  ⚠️  Bilinmeyen sınıf: '{name}' — {xml_path.name}")
            continue

        bbox = obj.find('bndbox')
        if bbox is None:
            continue

        xmin_el = bbox.find('xmin')
        ymin_el = bbox.find('ymin')
        xmax_el = bbox.find('xmax')
        ymax_el = bbox.find('ymax')

        if any(el is None or el.text is None 
               for el in [xmin_el, ymin_el, xmax_el, ymax_el]):
            continue

        xmin = float(xmin_el.text)  # type: ignore
        ymin = float(ymin_el.text)  # type: ignore
        xmax = float(xmax_el.text)  # type: ignore
        ymax = float(ymax_el.text)  # type: ignore

        cx, cy, w, h = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_txt.write_text('\n'.join(lines))

def find_annotation_dir(etiket_root: Path) -> Path | None:
    candidates = [
        etiket_root / 'THYZ_2025_Oturum_1' / 'Annotations' / 'Annotations',
        etiket_root / 'Annotations' / 'Annotations',
        etiket_root / 'Annotations',
    ]
    for c in candidates:
        if c.exists() and any(c.glob('*.xml')):
            return c
    # recursive ara
    for p in etiket_root.rglob('*.xml'):
        return p.parent
    return None

def find_image_dir(oturum_root: Path) -> Path | None:
    for candidate in [
        oturum_root / 'JPEGImages',
        oturum_root / 'images',
        oturum_root,
    ]:
        if candidate.exists():
            if any(candidate.glob('*.webp')) or any(candidate.glob('*.png')) \
            or any(candidate.glob('*.PNG')) or any(candidate.glob('*.jpg')):
                return candidate
    return None

def prepare_task1_dataset(
    raw_dir: str = 'data/raw',
    out_dir: str = 'data/datasets/task1',
    val_ratio: float = 0.15,
    seed: int = 42,
):
    raw_root = Path(raw_dir)
    out_root = Path(out_dir)

    # Çıktı klasörleri
    for split in ['train', 'val']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Her oturumu tara
    oturumlar = sorted([
        d for d in raw_root.iterdir()
        if d.is_dir()
        and 'Oturum' in d.name
        and 'etiket' not in d.name
        and 'Translation' not in d.name
    ])

    print(f"📁 {len(oturumlar)} oturum bulundu")

    for oturum_dir in oturumlar:
        num = oturum_dir.name.split('_')[-1]  # "1", "2", "3", "4"
        ann_dir = find_annotation_dir(oturum_dir)
        img_dir = find_image_dir(oturum_dir)

        if ann_dir is None:
            print(f"  ⚠️  Annotation bulunamadı: {etiket_dir.name}")
            continue

        xml_files = sorted(ann_dir.glob('*.xml'))
        print(f"  ✅ Oturum {num}: {len(xml_files)} annotation")

        for xml_file in xml_files:
            stem = xml_file.stem  # frame_000000
            img_file = None

            if img_dir:
                for ext in ['.PNG', '.png', '.JPG', '.jpg', '.webp', '.WEBP']:
                    c = img_dir / (stem + ext)
                    if c.exists():
                        img_file = c
                        break

            # Görüntü bulunamazsa sadece label üret (opsiyonel)
            all_pairs.append((img_file, xml_file))

    print(f"\n📊 Toplam: {len(all_pairs)} frame")

    # Train / Val böl
    random.seed(seed)
    random.shuffle(all_pairs)
    val_n = int(len(all_pairs) * val_ratio)
    splits = {
        'val':   all_pairs[:val_n],
        'train': all_pairs[val_n:],
    }

    for split, pairs in splits.items():
        print(f"  {split}: {len(pairs)} frame")
        for img_path, xml_path in tqdm(pairs, desc=f"  {split}"):
            # Label dönüştür
            dst_lbl = out_root / 'labels' / split / (xml_path.stem + '.txt')
            convert_xml_to_yolo(xml_path, dst_lbl)

            # Görüntüyü kopyala
            if img_path and img_path.exists():
                dst_img = out_root / 'images' / split / img_path.name
                shutil.copy2(img_path, dst_img)

    # dataset.yaml
    yaml_data = {
        'path': str(out_root.absolute()),
        'train': 'images/train',
        'val':   'images/val',
        'nc': 4,
        'names': {0: 'tasit', 1: 'insan', 2: 'uap', 3: 'uai'},
    }
    with open(out_root / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f, allow_unicode=True)

    print(f"\n✅ Dataset hazır → {out_root}")
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])}")
    print(f"   YAML:  {out_root / 'dataset.yaml'}")

if __name__ == '__main__':
    prepare_task1_dataset()