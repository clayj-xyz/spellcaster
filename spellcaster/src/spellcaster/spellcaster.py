from enum import StrEnum
from typing import Optional

from fire import Fire

from spellcaster.wand_tracker import WandTracker
from spellcaster.spell_handler import InferenceSpellHandler, TrainingSpellHandler
from spellcaster.utils import (
    Camera,
    WandPathVisualizer,
    CV2WandPathVisualizer,
    ShmWandPathVisualizer,
    ExitChecker,
    CV2ExitChecker,
    SigTermExitChecker,
    get_blob_detector
)


class Env(StrEnum):
    SUBPROCESS = "subprocess"
    STANDALONE = "standalone"


class Spellcaster:
    def __init__(
        self,
        camera: Camera,
        wand_tracker: WandTracker,
        exit_checker: ExitChecker,
        visualizer: Optional[WandPathVisualizer] = None
    ):
        self.camera = camera
        self.wand_tracker = wand_tracker
        self.exit_checker = exit_checker
        self.visualizer = visualizer
        
    def run(self, debug: bool = False):
        spell_handler = InferenceSpellHandler(debug=debug)
        self.wand_tracker.set_spell_handler(spell_handler)

        for frame in self.camera.stream():
            wand_path = self.wand_tracker.process_frame(frame)
            if debug:
                self.visualizer(frame, wand_path)
            if self.exit_checker.should_exit():
                break

    def train(self, spell_name: str):
        spell_handler = TrainingSpellHandler(spell_name)
        self.wand_tracker.set_spell_handler(spell_handler)

        for frame in self.camera.stream():
            wand_path = self.wand_tracker.process_frame(frame)
            self.visualizer(frame, wand_path)
            if self.exit_checker.should_exit():
                break

            
def build_spellcaster(env: Env):    
    camera = Camera()
    exit_checker = None
    visualizer = None

    match env:
        case Env.STANDALONE:
            exit_checker = CV2ExitChecker()
            visualizer = CV2WandPathVisualizer()
        case Env.SUBPROCESS:
            exit_checker = SigTermExitChecker()
            visualizer = ShmWandPathVisualizer()
        case _:
            raise ValueError(f"Unknown env: {env}")
        
    wand_tracker = WandTracker(get_blob_detector())
    return Spellcaster(camera, wand_tracker, exit_checker, visualizer)


def run(debug: bool = False, env: Env = Env.STANDALONE):
    spellcaster = build_spellcaster(env)
    spellcaster.run(debug)


def train(spell_name: str, env: Env = Env.STANDALONE):
    spellcaster = build_spellcaster(env)
    spellcaster.train(spell_name)


if __name__ == "__main__":
    Fire({
        "run": run,
        "train": train
    })