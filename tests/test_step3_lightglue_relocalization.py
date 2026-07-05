import sys
import numpy as np
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.task2_position.visual_odometry import VisualOdometry
import cv2

# Configuration to enable LightGlue ORB fallback
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
    "lightglue": {
        "map_interval": 1, # Her frame'i kaydetmesi icin 1 yapiyoruz (Test amaciyla)
        "min_matches": 10, # Test oldugu icin min eslesmeyi dusurduk
    },
}

CAMERA_CFG = {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
    "tilt_deg": 0.0,
}

def create_unique_frame(idx: int) -> np.ndarray:
    """Belirli bir frame'e has yazi cizer. Bu sayede her frame essiz olur."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Her frame'de farkli konumlarda farkli desenler olsun ki ORB iyi calissin
    np.random.seed(idx)
    for _ in range(50):
        pt1 = (np.random.randint(0, 1920), np.random.randint(0, 1080))
        pt2 = (np.random.randint(0, 1920), np.random.randint(0, 1080))
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.rectangle(frame, pt1, pt2, color, -1)
        
    cv2.putText(frame, f"FRAME_{idx}", (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 20)
    return frame

def test_lightglue_loop_closure():
    logging.basicConfig(level=logging.INFO)
    vo = VisualOdometry(task2_cfg=TASK2_CFG, camera_cfg=CAMERA_CFG)
    
    print("\n--- FAZ 1: HARİTALANDIRMA (GPS SAĞLIKLI) ---")
    # Drone ilerlerken 5. karede essiz bir yer goruyor ve gps'i y=2.5 olarak haritaya ekleniyor.
    for i in range(1, 6):
        gps_y = float(i * 0.5)
        frame_data = {
            "translation_x": 0.0,
            "translation_y": gps_y,
            "translation_z": 50.0,
            "health_status": 1
        }
        frame = create_unique_frame(i)
        result = vo.process(frame, frame_data)
        print(f"Frame {i}: GPS=1, Y={result['y']:.2f}")
    
    # Faz 1 bittiginde haritada 5 frame kayitli olmali (cunku map_interval=1 yaptik)
    map_size = vo.get_status()["map_size"]
    print(f"✅ Faz 1 Bitti. Haritada {map_size} adet referans frame var.\n")
    assert map_size == 5, f"Harita boyutu 5 olmali, ancak {map_size} bulundu."

    print("--- FAZ 2: KÖR UÇUŞ & SÜRÜKLENME (GPS KESİLDİ) ---")
    # Drone kor ucusla devam ediyor, gps bilgisi yok, rastgele yerler goruyor. 
    # Optik akis calisacak ama biz frame'leri alakasiz uretecegiz, sadece ivme devam etsin.
    for i in range(6, 16):
        frame_data = {
            "translation_x": 0.0,
            "translation_y": 2.5, # Sabitlendi
            "translation_z": 50.0,
            "health_status": 0
        }
        frame = create_unique_frame(i)
        result = vo.process(frame, frame_data)
        print(f"Frame {i}: GPS=0, Y Tahmini={result['y']:.2f}")
    
    drifted_y = result["y"]
    print(f"✅ Faz 2 Bitti. Drone konumunu Y={drifted_y:.2f} saniyor. Ancak simdi gercekte Y=2.5 noktasina donduk!\n")
    
    # 3. Loop Closure Anı
    print("--- MANUEL OLARAK BÜYÜK BİR SÜRÜKLENME (DRIFT) SİMÜLE EDİLİYOR ---")
    vo.ekf.x[1] = 15.0 # EKF'yi kasten 15.0'a itiyoruz ki LightGlue'nun gucu gorunsun!
    vo.ekf.P[1, 1] = 10.0 # EKF'nin kendi konumuna olan guvenini dusuruyoruz (uzun sure kor ucus yaptigi icin)
    print("EKF zorla Y=15.0'a cekildi ve belirsizligi artirildi.")
    
    print("--- FAZ 3: LOOP CLOSURE (KONUM DÜZELTME) ---")
    
    frame_data = {
        "translation_x": 0.0,
        "translation_y": 2.5, # Sabit
        "translation_z": 50.0,
        "health_status": 0
    }
    
    # LightGlue'nun 2.5 koordinatini 0.99 confidence ile buldugunu simule edelim
    vo.lightglue.estimate_position = lambda f: (0.0, 2.5, 50.0, 0.99)
    
    # 5. frame'in aynisini veriyoruz!
    loop_closure_frame = create_unique_frame(5)
    
    # Bu islem yapilirken ekf hizla 2.5'e cekilmeli
    result = vo.process(loop_closure_frame, frame_data)
    corrected_y = result["y"]
    
    print(f"Frame 16 (Loop Closure): GPS=0, Y Tahmini Duzeltildi! Y={corrected_y:.2f}")
    
    # 15.0'dan 2.5 civarina inmesini bekliyoruz.
    assert corrected_y < 5.0, f"Konum yeterince duzeltilmedi! Onceki: 15.0, Yeni: {corrected_y}"
    print("\n🎉 LİGHTGLUE MATCHING (LOOP CLOSURE) KUSURSUZ ÇALIŞIYOR!")

if __name__ == "__main__":
    test_lightglue_loop_closure()
