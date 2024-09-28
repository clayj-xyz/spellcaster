from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .actions_registry import actions
from .constants import FRAME_SHAPE, DATA_DIR, MODEL_PATH
from .db import get_spell, get_action
from .modeling.train import BasicConvNet
from .modeling.data_loader import CropWandPath


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
        self.model = BasicConvNet()
        stored_model = torch.load(MODEL_PATH)
        self.model.load_state_dict(stored_model["state_dict"])
        self.model.eval()
        self.class_to_spell = {
            cls: spell for cls, spell in enumerate(stored_model["classes"])
        }
        
    def handle_spell(self, wand_path: list[tuple[int, int]]):
        wand_path_img = self.draw_wand_path(wand_path)
        cropped_wand_path_img = CropWandPath()(wand_path_img)
        wand_path_img_tensor = torch.tensor(cropped_wand_path_img)[None, None, ...].float()
        resized_wand_path = torch.nn.functional.interpolate(wand_path_img_tensor, size=(28, 28))

        with torch.no_grad():
            y_pred = self.model(resized_wand_path)
            spell_name = self.class_to_spell[torch.argmax(y_pred).item()]
            print(f"spell detected: {spell_name}")
            spell = get_spell(spell_name)
            if spell.action_id is not None:
                action = get_action(spell.action_id)
                print(f"executing action: {action.name}")
                actions[action.function]()
            


class TrainingSpellHandler(SpellHandler):
    def __init__(self, spell_name: str, spell_handled_callback=None):
        self.images_dir = Path(DATA_DIR) / "images" / spell_name
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.spell_handled_callback = spell_handled_callback

    def handle_spell(self, wand_path: list[tuple[int, int]]):
        self.wand_path_img = self.draw_wand_path(wand_path)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wand_path_img_path = self.images_dir / f"{current_time}.png"
        cv2.imwrite(str(wand_path_img_path), self.wand_path_img)
        print(f"saved wand path image to {wand_path_img_path}")
        if self.spell_handled_callback is not None:
            self.spell_handled_callback()
