import time
from enum import StrEnum
from typing import Optional

from fire import Fire

from .db import get_spell, add_spell, manage_db, Spell
from .wand_tracker import WandTracker
from .spell_handler import InferenceSpellHandler, TrainingSpellHandler
from .modeling.train import train
from .utils import (
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

    def collect_training_data(self, spell_name: str, num_samples: int = 15):
        if get_spell(spell_name) is None:
            add_spell(Spell(name=spell_name))

        should_reset = False
        def reset():
            nonlocal should_reset
            should_reset = True
        spell_handler = TrainingSpellHandler(spell_name, spell_handled_callback=reset)
        self.wand_tracker.set_spell_handler(spell_handler)

        for _ in range(num_samples):
            for i in range(3):
                time.sleep(1)
                print(f"starting in {3 - i}", end="\r")
            print("\r", end="")

            early_exit = False
            should_reset = False
            for frame in self.camera.stream():
                wand_path = self.wand_tracker.process_frame(frame)
                self.visualizer(frame, wand_path)
                early_exit = self.exit_checker.should_exit()
                if early_exit or should_reset:
                    break

            if early_exit:
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


def collect_training_data(spell_name: str, env: Env = Env.STANDALONE):
    spellcaster = build_spellcaster(env)
    spellcaster.collect_training_data(spell_name)


def cli():
    Fire({
        "run": run,
        "collect_training_data": collect_training_data,
        "manage": manage_db,
        "train_model": train
    })


if __name__ == "__main__":
    cli()