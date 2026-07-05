import sys
import time
import logging
from pathlib import Path
import threading
import multiprocessing
import subprocess
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.communication.api_client import CompetitionAPIClient
from src.communication.frame_fetcher import FrameFetcher
from src.communication.result_sender import ResultSender

def run_server():
    from tests.mock_server import create_app
    from aiohttp import web
    app = create_app()
    web.run_app(app, host='127.0.0.1', port=5000, handle_signals=False, print=None)

def test_api_flow():
    # 1. Start mock server in the background
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    
    try:
        # Give server a second to start
        time.sleep(2)
        
        # 2. Initialize API Client
        api = CompetitionAPIClient(
            server_url="http://127.0.0.1:5000",
            token="TEST_TOKEN",
            timeout=2,
            cls_as_url=True
        )
        
        print("Testing Connection...")
        assert api.test_connection() == True, "Connection test failed!"
        print("✅ Connection OK")
        
        # 3. Get frames
        frames = api.get_frame_list()
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"
        print(f"✅ Fetched {len(frames)} frames")
        
        # 4. Fetch first frame image
        fetcher = FrameFetcher(api_client=api, buffer_size=2)
        first_frame = frames[0]
        fetched_data = fetcher.fetch(first_frame)
        
        assert fetched_data is not None, "Failed to fetch image data"
        assert "frame" in fetched_data, "Missing frame in fetched data"
        assert fetched_data["frame"].shape == (1080, 1920, 3), "Invalid frame shape"
        print("✅ Image fetch and decode OK")
        
        # 5. Send a dummy result
        sender = ResultSender(api_client=api, max_retries=1)
        
        dummy_result = {
            "detections": [
                {"class_id": 1, "landing_status": -1, "motion_status": -1, "bbox": [10.0, 20.0, 30.0, 40.0], "confidence": 0.9}
            ],
            "position": {"x": 1.5, "y": 2.5, "z": -0.5},
            "matched_objects": []
        }
        
        success = sender.send(first_frame["url"], dummy_result)
        assert success == True, "Result sending failed!"
        print("✅ Result packing and sending OK")
        
        print("\n🎉 BÜTÜN TESTLER BAŞARIYLA TAMAMLANDI! (API Akışı Sorunsuz)")
        
    finally:
        server_process.terminate()
        server_process.join()

if __name__ == "__main__":
    test_api_flow()
