import math

import cv2
import numpy as np

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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    blob_detector = get_blob_detector()
    wand_tracker = WandTracker(blob_detector)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        
        wand_path = wand_tracker.process_frame(frame)
        wand_path_img = draw_wand_path(frame, wand_path)
        cv2.imshow('wand path', wand_path_img)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()