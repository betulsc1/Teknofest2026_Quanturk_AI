import sys
import numpy as np
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.task2_position.visual_odometry import VisualOdometry
import cv2

# Basic config mimicking the actual setup
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
    # Disable RAFT for test speed, use Farneback
    "use_raft": False,
    # "lightglue": {}, # Disabled so it doesn't pull position back in synthetic test
}

CAMERA_CFG = {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
    "tilt_deg": 0.0,
}


def test_state_machine():
    logging.basicConfig(level=logging.DEBUG)
    vo = VisualOdometry(task2_cfg=TASK2_CFG, camera_cfg=CAMERA_CFG)
    
    # 1. Gölge Modu (GPS Healthy) - Frame 1-10
    print("\n--- FAZ 1: GPS SAĞLIKLI (Shadow Mode) ---")
    base_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create a noisy background (checkerboard or random) so optical flow has global motion
    # Create a background with random circles so Farneback has unique features
    np.random.seed(42)
    base_bg = np.zeros((2000, 2000, 3), dtype=np.uint8)
    for _ in range(500):
        cx = np.random.randint(0, 2000)
        cy = np.random.randint(0, 2000)
        r = np.random.randint(10, 100)
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.circle(base_bg, (cx, cy), r, color, -1)
    
    last_z = 0
    for i in range(1, 16):
        # Drone moving forward in Y axis
        gps_x, gps_y, gps_z = 0.0, float(i * 0.5), 50.0 
        
        frame_data = {
            "translation_x": gps_x,
            "translation_y": gps_y,
            "translation_z": gps_z,
            "health_status": 1
        }
        
        # Crop the 1920x1080 window from base_bg, shifting it
        y_shift = i * 10
        frame = base_bg[y_shift:y_shift+1080, 0:1920].copy()
        
        result = vo.process(frame, frame_data)
        
        print(f"Frame {i}: GPS=1, Y Tahmini={result['y']:.2f}")
        assert result["gps_healthy"] == True, "State Machine failed: GPS should be healthy"
        # Since EKF smooths, it should be close to gps_y
        assert abs(result["y"] - gps_y) < 1.5, f"Y diff too large: {result['y']} vs {gps_y}"
        
        last_z = result["z"]

    last_healthy_y = result["y"]
    print(f"✅ Faz 1 Tamamlandı. Son Konum Y: {last_healthy_y:.2f}")

    # Disable LightGlue so it doesn't snap position back to old keyframes during the synthetic test
    vo.lightglue.estimate_position = lambda f: None

    # 2. Kör Uçuş Modu (Dead Reckoning) - Frame 16-25
    print("\n--- FAZ 2: GPS KESİLDİ (Dead Reckoning Mode) ---")
    
    for i in range(16, 26):
        frame_data = {
            "translation_x": 0.0,
            "translation_y": 5.0, # Stuck at last value
            "translation_z": 50.0,
            "health_status": 0
        }
        
        # Continue shifting the background by 10 pixels
        y_shift = i * 10
        frame = base_bg[y_shift:y_shift+1080, 0:1920].copy()
        
        result = vo.process(frame, frame_data)
        
        print(f"Frame {i}: GPS=0, Y Tahmini={result['y']:.2f}")
        assert result["gps_healthy"] == False, "State Machine failed: GPS should be unhealthy"
        assert result["y"] > last_healthy_y, "Dead reckoning failed: Position did not advance without GPS"
        
        last_healthy_y = result["y"]

    print(f"✅ Faz 2 Tamamlandı. VO Konum Y: {last_healthy_y:.2f}")
    print("\n🎉 GPS SAĞLIK DURUMU GEÇİŞ MANTIĞI KUSURSUZ ÇALIŞIYOR!")


if __name__ == "__main__":
    test_state_machine()
