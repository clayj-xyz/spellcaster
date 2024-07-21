from enum import Enum
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


class SpellcasterMode(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"
    DEBUG = "debug"
    STANDBY = "standby"


class RunMode(str, Enum):
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
        
    def run(self):
        for frame in self.camera.stream():
            wand_path = self.wand_tracker.process_frame(frame)
            if self.visualizer is not None:
                self.visualizer(frame, wand_path)
            if self.exit_checker.should_exit():
                break

            
def build_spellcaster(run_mode: RunMode, spellcaster_mode: SpellcasterMode):
    if spellcaster_mode == SpellcasterMode.STANDBY:
        raise ValueError("cannot build spellcaster in standby mode")
    
    camera = Camera()
    spell_handler = None
    exit_checker = None
    visualizer = None

    match run_mode:
        case RunMode.STANDALONE:
            exit_checker = CV2ExitChecker()
            visualizer = CV2WandPathVisualizer()
        case RunMode.SUBPROCESS:
            exit_checker = SigTermExitChecker()
            if spellcaster_mode in (SpellcasterMode.DEBUG, SpellcasterMode.TRAINING):
                visualizer = ShmWandPathVisualizer()
        case _:
            raise ValueError(f"Unknown run mode: {run_mode}")

    match spellcaster_mode:
        case SpellcasterMode.INFERENCE:
            spell_handler = InferenceSpellHandler()
        case SpellcasterMode.DEBUG:
            spell_handler = InferenceSpellHandler(debug=True)
        case SpellcasterMode.TRAINING:
            spell_handler = TrainingSpellHandler()
        case _:
            raise ValueError(f"Unknown spellcaster mode: {spellcaster_mode}")
        
    wand_tracker = WandTracker(get_blob_detector(), spell_handler)
    return Spellcaster(camera, wand_tracker, exit_checker, visualizer)


def main(spellcaster_mode: SpellcasterMode):
    spellcaster = build_spellcaster(RunMode.STANDALONE, spellcaster_mode)
    spellcaster.run()


if __name__ == "__main__":
    Fire(main)