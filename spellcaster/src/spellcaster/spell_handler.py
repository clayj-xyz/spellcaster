from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from spellcaster.constants import FRAME_SHAPE, IMAGES_DIR


class SpellHandler(ABC):
    def draw_wand_path(self, wand_path: list[tuple[int, int]]):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        for i in range(1, len(wand_path)):
            thickness = int(np.sqrt(len(wand_path) / float(i + 1)) * 2.5)
            cv2.line(frame, wand_path[i - 1], wand_path[i], (255, 255, 255), thickness)
        
        frame = cv2.resize(frame, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @abstractmethod
    def handle_spell(self, wand_path: list[tuple[int, int]]):
        pass


class InferenceSpellHandler(SpellHandler):
    def __init__(self, debug: bool=False):
        self.debug = debug
        
    def handle_spell(self, wand_path: list[tuple[int, int]]):
        print("spell detected")


class TrainingSpellHandler(SpellHandler):
    def __init__(self, spell_name: str):
        self.images_dir = Path(IMAGES_DIR) / spell_name
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def handle_spell(self, wand_path: list[tuple[int, int]]):
        self.wand_path_img = self.draw_wand_path(wand_path)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wand_path_img_path = self.images_dir / f"{current_time}.png"
        cv2.imwrite(str(wand_path_img_path), self.wand_path_img)
        print(f"saved wand path image to {wand_path_img_path}")
