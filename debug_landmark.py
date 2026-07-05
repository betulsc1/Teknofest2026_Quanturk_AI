import json, cv2
from src.task2_position.visual_odometry import VisualOdometry
import logging
logging.basicConfig(level=logging.DEBUG)

vo = VisualOdometry({})
data = json.load(open('data/datasets/task2/ground_truth.json'))

for fdata in data['frames'][590:610]:
    frame = cv2.imread(f"data/datasets/task2/frames/{fdata['filename']}")
    detections = []
    from src.task2_position.frame_processor import Detection
    for d in fdata.get('detections', []):
        detections.append(Detection(**d))
    frame_data = {
        "translation_x": fdata["translation_x"],
        "translation_y": fdata["translation_y"],
        "translation_z": fdata["translation_z"],
        "health_status": fdata["health_status"]
    }
    res = vo.process(frame, frame_data, detections)
    print(f"Frame {fdata['frame_id']} - Error: {((res['x']-fdata['translation_x'])**2 + (res['y']-fdata['translation_y'])**2)**0.5:.2f}m")
