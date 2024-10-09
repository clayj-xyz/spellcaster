import time
import asyncio
import threading

import cv2

from spellcaster.utils.shared_buffer import SharedFrameBufferReader
from spellcaster.constants import FRAME_RATE


_latest_frame = None
def capture_stream(stop_event: threading.Event):
    global _latest_frame
    shared_frame_buffer = SharedFrameBufferReader()
    for frame in shared_frame_buffer.read():
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            _latest_frame = jpeg.tobytes()

        if stop_event.is_set():
            break

        time.sleep(1 / FRAME_RATE)


class SpellcasterViewer:
    def __init__(self):
        self.streams = 0
        self.stop_event = threading.Event()
        self.init_thread()

    def init_thread(self):
        self.livestream_thread = threading.Thread(
            target=capture_stream, args=(self.stop_event,)
        )
        self.stop_event.clear()
        
    async def get_stream(self, request):
        self.start_livestream()
        try:
            while True:
                if _latest_frame is not None:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + _latest_frame + b'\r\n')
                await asyncio.sleep(1 / FRAME_RATE)
        except asyncio.CancelledError:
            self.stop_livestream()

    def start_livestream(self):
        if self.streams == 0:
            self.livestream_thread.start()
        self.streams += 1

    def stop_livestream(self):
        self.streams -= 1
        if self.streams == 0:
            self.stop_event.set()
            self.livestream_thread.join()
            self.init_thread()

    def __del__(self):
        self.stop_livestream()
        
