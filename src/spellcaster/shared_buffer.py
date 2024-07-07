from multiprocessing.shared_memory import SharedMemory

import numpy as np

from spellcaster.constants import FRAME_SHAPE

DEFAULT_FRAME_BUFFER_NAME = "frame_buffer"


class SharedFrameBufferWriter:
    def __init__(self, name=DEFAULT_FRAME_BUFFER_NAME, shape=FRAME_SHAPE):
        ex_frame = np.zeros(shape, dtype=np.uint8)
        self.shm = SharedMemory(name=name, create=True, size=ex_frame.nbytes)
        self.buffer = np.ndarray(shape, buffer=self.shm.buf, dtype=np.uint8)

    def write(self, frame):
        self.buffer[:] = frame[:]

    def __del__(self):
        self.shm.close()
        self.shm.unlink()
    

class SharedFrameBufferReader:
    def __init__(self, name=DEFAULT_FRAME_BUFFER_NAME, shape=FRAME_SHAPE):
        self.shm = SharedMemory(name=name)
        self.buffer = np.ndarray(shape, buffer=self.shm.buf, dtype=np.uint8)

    def read(self):
        while True:
            yield self.buffer

    def __del__(self):
        self.shm.close()