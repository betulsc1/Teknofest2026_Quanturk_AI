import asyncio
import json
import logging
from aiohttp import web
import numpy as np
import cv2

logger = logging.getLogger("mock_server")

routes = web.RouteTableDef()

# Dummy image cache
DUMMY_IMAGE = None

def get_dummy_image():
    global DUMMY_IMAGE
    if DUMMY_IMAGE is None:
        # Create a 1920x1080 black image
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        DUMMY_IMAGE = buffer.tobytes()
    return DUMMY_IMAGE


@routes.get('/frames/')
async def get_frames(request):
    logger.info("GET /frames/ requested")
    
    # Simulate a small competition session with 10 frames
    frames = []
    for i in range(1, 11):
        # First 5 frames healthy, next 5 unhealthy
        health = 1 if i <= 5 else 0
        frames.append({
            "url": f"http://127.0.0.1:5000/frames/{i}/",
            "image_url": f"/images/{i}.jpg",
            "video_name": "test_video_1",
            "session": "http://127.0.0.1:5000/session/1/",
            "translation_x": 0.05 * i,
            "translation_y": 0.01 * i,
            "translation_z": -0.02 * i,
            "health_status": health,
            # Let's also include gps_health_status in case it uses that
            "gps_health_status": health 
        })
    return web.json_response(frames)


@routes.get('/images/{id}.jpg')
async def get_image(request):
    return web.Response(body=get_dummy_image(), content_type='image/jpeg')


@routes.get('/reference-objects/')
async def get_references(request):
    logger.info("GET /reference-objects/ requested")
    return web.json_response([])


@routes.post('/results/')
async def post_results(request):
    try:
        data = await request.json()
        logger.info(f"POST /results/ received for frame: {data.get('frame')}")
        
        # Simple validation based on Sekil 17
        required_keys = ["frame", "detected_objects", "detected_translations", "detected_undefined_objects"]
        for key in required_keys:
            if key not in data:
                return web.json_response({"error": f"Missing key: {key}"}, status=400)
                
        return web.json_response({"status": "success"})
    except Exception as e:
        logger.error(f"Error parsing POST /results/: {e}")
        return web.json_response({"error": str(e)}, status=400)


def create_app():
    app = web.Application()
    app.add_routes(routes)
    return app

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    web.run_app(app, host='127.0.0.1', port=5000)
