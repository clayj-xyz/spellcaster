import math

import cv2

from spellcaster.spell_handler import SpellHandler


class WandTracker:
    def __init__(
        self,
        blob_detector: cv2.SimpleBlobDetector,
        spell_handler: SpellHandler
    ):
        self.blob_detector = blob_detector
        self.spell_handler = spell_handler
        self.wand_path = []
        self.minimum_wand_path_len = 30
        self.patience = 15
        self.empty_frame_cnt = 0

    def get_wand_keypoint(self, keypoints):
        if len(keypoints) == 0:
            return None
        if len(self.wand_path) == 0:
            return keypoints[0]
        
        possible_wand_keypoints = []
        for k in keypoints:
            dist = math.dist(k, self.wand_path[-1])
            if dist > 5 and dist < 100:
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
                    self.spell_handler.handle_spell(self.wand_path)
                self.wand_path.clear()
        else:
            self.empty_frame_cnt = 0
            self.wand_path.append(wand_keypoint)

        return self.wand_path
    