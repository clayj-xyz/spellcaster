import time
import threading

import cv2
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from spellcaster.shared_buffer import SharedFrameBufferReader
from spellcaster.constants import FRAME_RATE


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

    
latest_frame = None
def capture_stream():
    global latest_frame
    shared_frame_buffer = SharedFrameBufferReader()
    for frame in shared_frame_buffer.read():
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()
        time.sleep(1 / FRAME_RATE)  # Adjust to control the frame rate

# Start the capture thread
capture_thread = threading.Thread(target=capture_stream)
capture_thread.daemon = True
capture_thread.start()

@app.get("/stream")
def stream():
    def video_streamer():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(1 / FRAME_RATE)

    return StreamingResponse(video_streamer(), media_type='multipart/x-mixed-replace; boundary=frame')
