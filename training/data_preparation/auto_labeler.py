import cv2
import os
import sys

# Scriptin src klasörünü bulabilmesi için proje ana dizinini path'e ekliyoruz
# training/data_preparation içinden iki üst klasöre (ana dizine) çıkıyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.task1_detection.detector import ObjectDetector

def process_video(video_path, output_base_dir, detector_cfg, fps_extract=1, prefix=""):
    # Resimler ve etiketler için alt klasörleri oluştur (İzolasyon klasöründe)
    img_dir = os.path.join(output_base_dir, "images")
    lbl_dir = os.path.join(output_base_dir, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    print(f"Model yükleniyor... ({prefix})")
    detector = ObjectDetector(detector_cfg)
    # Küçük nesneleri yakalamak için SAHI aktif
    detector.enable_sahi(slice_size=640, overlap=0.2)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = round(cap.get(cv2.CAP_PROP_FPS))
    if video_fps == 0: video_fps = 30
    
    frame_interval = max(1, video_fps // fps_extract)
    count = 0
    saved = 0
    
    print(f"Video işleniyor: {video_path} | {fps_extract} FPS ile kare alınacak.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            img_h, img_w = frame.shape[:2]
            detections = detector.detect(frame)
            
            # İsim çakışmasını önlemek için prefix ekliyoruz (örn: rgb_frame_00001)
            base_name = f"{prefix}frame_{saved:05d}"
            
            img_path = os.path.join(img_dir, f"{base_name}.jpg")
            txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            cv2.imwrite(img_path, frame)
            
            with open(txt_path, "w") as f:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cls_id = det["class_id"]
                    
                    # YOLO Normalize Formatı (0-1 arası)
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    xc = ((x1 + x2) / 2.0) / img_w
                    yc = ((y1 + y2) / 2.0) / img_h
                    
                    # Sınır dışına taşmaları önlemek için kilitleme
                    xc = max(0.0, min(xc, 1.0))
                    yc = max(0.0, min(yc, 1.0))
                    w = max(0.0, min(w, 1.0))
                    h = max(0.0, min(h, 1.0))
                    
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            saved += 1
            if saved % 50 == 0:
                print(f"{saved} kare etiketlendi ve {img_dir} klasörüne kaydedildi...")
                
        count += 1
        
    cap.release()
    print(f"BİTTİ! Toplam {saved} kare {output_base_dir} konumuna çıkarıldı.\n")

if __name__ == "__main__":
    # detector.py için güvenli etiketleme konfigürasyonu
    cfg = {
        "model_path": "models/task1/detector/best.pt", 
        "confidence": 0.25,  # Etiketlemede düşük tutuyoruz ki her şeyi bulsun
        "iou_threshold": 0.45,
        "device": "cuda:0"
    }
    
    # İnen videoların bilgisayarınızdaki tam yolunu veya proje ana dizinine göre yolunu yazın
    video_rgb = "THYZ_2026_Ornek_Veri_1_RGB.mp4" 
    video_termal = "THYZ_2026_Ornek_Veri_2_Termal.mp4"
    
    # HEDEF KLASÖR: Sadece yeni veriler için "2026_raw" karantina klasörü
    dataset_dir = os.path.join(project_root, "data", "datasets", "2026_raw")
    
    # Sırayla çalıştırıyoruz
    process_video(video_rgb, dataset_dir, cfg, fps_extract=1, prefix="rgb_")
    process_video(video_termal, dataset_dir, cfg, fps_extract=1, prefix="termal_")
