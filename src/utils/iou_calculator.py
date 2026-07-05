"""
src/utils/iou_calculator.py
IoU ve WBF (Weighted Box Fusion) hesaplamaları.
Şartname: IoU eşik değeri 0.5
"""
import numpy as np
from src.utils.bbox_utils import bbox_area

def calculate_iou(bbox1: list, bbox2: list) -> float:
    """
    İki bounding box arasındaki IoU'yu hesaplar.

    Şartname Denklem 1:
        IoU = (GerçekReferansDörtgen ∩ TahminEdilenDörtgen)
              / (GerçekReferansDörtgen ∪ TahminEdilenDörtgen)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Kesişim alanı
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    # Birleşim alanı
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms(detections: list,
        iou_threshold: float = 0.45) -> list:
    """
    Non-Maximum Suppression (NMS).
    Aynı nesne için birden fazla box varsa en yüksek güvenli olanı tutar.

    detections: [{"bbox": [...], "confidence": float, ...}, ...]
    """
    if not detections:
        return []

    # Güvene göre azalan sırala
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if calculate_iou(best["bbox"], d["bbox"]) < iou_threshold
        ]

    return kept


def weighted_box_fusion(boxes_list: list,
                        scores_list: list,
                        labels_list: list,
                        iou_thr: float = 0.55,
                        skip_box_thr: float = 0.3) -> tuple:
    """
    WBF (Weighted Box Fusion) — ensemble için.
    Birden fazla modelin çıktısını birleştirir.
    NMS'den üstün: box'ları siler değil, ağırlıklı ortalamasını alır.

    boxes_list  : her model için [[x1,y1,x2,y2], ...] listesi
    scores_list : her model için [conf, ...] listesi
    labels_list : her model için [class_id, ...] listesi

    Döndürür: (birleşik_boxes, birleşik_scores, birleşik_labels)
    """
    # Basit WBF implementasyonu
    all_boxes, all_scores, all_labels = [], [], []

    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes, scores, labels):
            if score >= skip_box_thr:
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(label)

    if not all_boxes:
        return [], [], []

    # Gruplama: IoU > iou_thr olanları birleştir
    used = [False] * len(all_boxes)
    result_boxes, result_scores, result_labels = [], [], []

    for i in range(len(all_boxes)):
        if used[i]:
            continue

        cluster_boxes  = [all_boxes[i]]
        cluster_scores = [all_scores[i]]
        cluster_label  = all_labels[i]
        used[i] = True

        for j in range(i + 1, len(all_boxes)):
            if not used[j] and all_labels[j] == cluster_label:
                if calculate_iou(all_boxes[i], all_boxes[j]) >= iou_thr:
                    cluster_boxes.append(all_boxes[j])
                    cluster_scores.append(all_scores[j])
                    used[j] = True

        # Ağırlıklı ortalama box
        weights = np.array(cluster_scores)
        weights = weights / weights.sum()
        fused = np.average(np.array(cluster_boxes), axis=0, weights=weights)

        result_boxes.append(fused.tolist())
        result_scores.append(float(np.mean(cluster_scores)))
        result_labels.append(cluster_label)

    return result_boxes, result_scores, result_labels