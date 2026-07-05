import sys
import numpy as np
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.task2_position.visual_odometry import VisualOdometry
from src.task2_position.landmark_manager import Detection
import cv2

TASK2_CFG = {
    "dt": 0.133,
    "ekf": {
        "q_pos": 0.01,
        "q_vel": 0.5,
        "r_gps": 0.05,
        "r_vo":  0.8,
        "dt": 0.133
    },
    "keyframe_interval": 50,
    "pose_window_size": 10,
    "drift_threshold": 2.0,
    "use_raft": False,
    "lightglue": {}, # Testi izole etmek icin kapattik
}

CAMERA_CFG = {
    "focal_length": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
}

def create_blank_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

def test_landmark_integration():
    logging.basicConfig(level=logging.INFO)
    vo = VisualOdometry(task2_cfg=TASK2_CFG, camera_cfg=CAMERA_CFG)
    vo.lightglue.estimate_position = lambda f: None # Sadece LandmarkManager'i test edecegiz
    
    print("\n--- FAZ 1: HEDEF TAKİBİ (GPS SAĞLIKLI) ---")
    # Drone ilerlerken bir UAP (Ucan Arac Platformu) tespit ediliyor.
    # UAP gercekte Y=10.0 konumunda havada asili duruyor olsun.
    # Drone Y=0.0'dan Y=2.0'a dogru hareket edecek.
    
    target_gps_y = 10.0
    scale = 50.0 / 1000.0 # 0.05 m/px
    
    for i in range(1, 6):
        drone_y = float(i * 1.0) # 1.0, 2.0, 3.0, 4.0, 5.0
        
        # drone_y = target_gps_y - delta_y
        # delta_y = (bbox_y - cy) * scale
        # (target_gps_y - drone_y) / scale = bbox_y - cy
        # bbox_y = cy + (target_gps_y - drone_y) / scale
        bbox_y = 540.0 + (target_gps_y - drone_y) / scale
        
        frame_data = {
            "translation_x": 0.0,
            "translation_y": drone_y,
            "translation_z": 50.0,
            "health_status": 1
        }
        
        # Gorev 1'den gelen hayali bir tespit
        detections = [
            Detection(class_name="UAP", confidence=0.95, bbox_x=960.0, bbox_y=bbox_y, bbox_w=50.0, bbox_h=50.0)
        ]
        
        result = vo.process(create_blank_frame(), frame_data, detections=detections)
        print(f"Frame {i}: GPS=1, Drone Y={result['y']:.2f}, UAP Pixel Y={bbox_y:.1f}")
        
    stats = vo.landmarks.get_stats()
    print(f"✅ Faz 1 Bitti. Güvenilir Landmark Sayısı: {stats['reliable']}\n")
    assert stats['reliable'] >= 1, "Landmark yeterince gorulmesine ragmen guvenilir (reliable) olamadi!"

    print("--- FAZ 2: KÖR UÇUŞ & LANDMARK ÜZERİNDEN KONUM TAHMİNİ (GPS KESİLDİ) ---")
    # GPS koptu. Drone gercekte ilerlemeye devam ediyor.
    # Diyelim ki Drone gercekte Y=8.0 konumuna geldi. Ama optik akis olmadigi icin EKF hala 5.0 saniyor.
    vo.ekf.x[1] = 5.0 # Kasten sabitliyoruz
    vo.ekf.P[1, 1] = 10.0 # Guvensiz
    
    true_drone_y = 8.0
    # Kamerada UAP yeni bir pikselde gorunmeli:
    new_bbox_y = 540.0 + (target_gps_y - true_drone_y) / scale
    
    frame_data = {
        "translation_x": 0.0,
        "translation_y": 5.0, # Sabit
        "translation_z": 50.0,
        "health_status": 0
    }
    
    detections = [
        Detection(class_name="UAP", confidence=0.98, bbox_x=960.0, bbox_y=new_bbox_y, bbox_w=55.0, bbox_h=55.0)
    ]
    
    print(f"EKF su an Y=5.0'da oldugunu saniyor. Kamerada UAP Y={new_bbox_y:.1f} pikselinde goruldu.")
    print(f"Gercekte Drone Y={true_drone_y:.2f} konumunda olmali.")
    
    result = vo.process(create_blank_frame(), frame_data, detections=detections)
    corrected_y = result["y"]
    
    print(f"Frame 6: GPS=0, Y Tahmini Duzeltildi! Y={corrected_y:.2f}")
    
    # 5.0'dan 8.0'a dogru ciddi bir ziplama yapmis olmasi gerekiyor
    assert corrected_y > 6.0, f"Landmark integrasyonu calismadi! Onceki: 5.0, Beklenen: ~8.0, Yeni: {corrected_y}"
    print("\n🎉 LANDMARK ÜZERİNDEN REVERSE PROJECTION KUSURSUZ ÇALIŞIYOR!")

if __name__ == "__main__":
    test_landmark_integration()
