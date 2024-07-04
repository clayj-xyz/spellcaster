import asyncio
import time
import threading
from typing import Union

import cv2
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from wand_tracker import read_wand_tracker


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


# @app.get("/activate")
# def activate(request: Request):
    
latest_frame = None
def capture_stream():
    global latest_frame
    for frame in read_wand_tracker():
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = jpeg.tobytes()
        time.sleep(0.05)  # Adjust to control the frame rate

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
            time.sleep(0.05)

    return StreamingResponse(video_streamer(), media_type='multipart/x-mixed-replace; boundary=frame')
