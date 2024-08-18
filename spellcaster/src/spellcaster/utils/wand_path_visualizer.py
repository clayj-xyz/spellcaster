from abc import ABC, abstractmethod

import cv2
import numpy as np

from .shared_buffer import SharedFrameBufferWriter


class WandPathVisualizer(ABC):
    def draw_wand_path(self, frame, wand_path):
        for i in range(1, len(wand_path)):
            thickness = int(np.sqrt(len(wand_path) / float(i + 1)) * 2.5)
            cv2.line(frame, wand_path[i - 1], wand_path[i], (0, 0, 255), thickness)
        return frame

    @abstractmethod
    def __call__(self, frame, wand_path):
        pass


class CV2WandPathVisualizer(WandPathVisualizer):
    def __call__(self, frame,wand_path):
        wand_path_frame = self.draw_wand_path(frame, wand_path)
        cv2.imshow('wand path', wand_path_frame)
        

class ShmWandPathVisualizer(WandPathVisualizer):
    def __init__(self):
        self.shared_frame_buffer = SharedFrameBufferWriter()
        
    def __call__(self, frame, wand_path):
        wand_path_img = self.draw_wand_path(frame, wand_path)
        self.shared_frame_buffer.write(wand_path_img)