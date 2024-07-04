import math
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from camera import Camera


class WandTracker:
    def __init__(self, blob_detector):
        self.blob_detector = blob_detector
        self.wand_path = []
        self.minimum_wand_path_len = 30
        self.patience = 30
        self.empty_frame_cnt = 0

    def get_wand_keypoint(self, keypoints):
        if len(keypoints) == 0:
            return None
        if len(self.wand_path) == 0:
            return keypoints[0]
        
        possible_wand_keypoints = []
        for k in keypoints:
            dist = math.dist(k, self.wand_path[-1])
            if dist > 1 and dist < 100:
                possible_wand_keypoints.append((k, dist))
        
        if len(possible_wand_keypoints) == 0:
            return None
        return min(possible_wand_keypoints, key=lambda t: t[1])[0]

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = self.blob_detector.detect(frame)
        keypoint_coords = [[round(coord) for coord in k.pt] for k in keypoints]
        wand_keypoint = self.get_wand_keypoint(keypoint_coords)
        if wand_keypoint is None:
            self.empty_frame_cnt += 1
            if self.empty_frame_cnt > self.patience:
                if len(self.wand_path) >= self.minimum_wand_path_len:
                    print("spell detected")
                self.wand_path.clear()
        else:
            self.wand_path.append(wand_keypoint)

        return self.wand_path


def get_blob_detector():
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255

    params.minThreshold = 150
    params.maxThreshold = 255

    params.filterByArea = True
    params.maxArea = 1000
    
    return cv2.SimpleBlobDetector_create(params)


def draw_wand_path(frame, wand_path):
    for i in range(1, len(wand_path)):
        thickness = int(np.sqrt(len(wand_path) / float(i + 1)) * 2.5)
        cv2.line(frame, wand_path[i - 1], wand_path[i], (0, 0, 255), thickness)
    return frame


def run_wand_tracker():
    camera = Camera()
    blob_detector = get_blob_detector()
    wand_tracker = WandTracker(blob_detector)

    for frame in camera.stream():
        yield frame, wand_tracker.process_frame(frame)


def serve_wand_tracker():
    print("starting wand tracker")
    wand_tracker = run_wand_tracker()
    frame, _ = next(wand_tracker)

    #create the shared memory for the frame buffer
    frame_buffer_shm = SharedMemory(name="wand_frame_buffer", create=True, size=frame.nbytes)
    frame_buffer = np.ndarray((480, 640, 3), buffer=frame_buffer_shm.buf, dtype=frame.dtype)

    for frame, wand_path in wand_tracker:
        frame_buffer[:] = draw_wand_path(frame, wand_path)[:]

    frame_buffer_shm.close()


def read_wand_tracker():
    frame_buffer_shm = SharedMemory(name="wand_frame_buffer")
    #create the framebuffer using the shm's memory
    frame_buffer = np.ndarray((480, 640, 3), buffer=frame_buffer_shm.buf, dtype='u1')
    while True:
        yield frame_buffer


def main():
    for frame, wand_path in run_wand_tracker():
        wand_path_frame = draw_wand_path(frame, wand_path)
        cv2.imshow('wand path', wand_path_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    #main()
    serve_wand_tracker()