from enum import StrEnum

from multiprocessing import Process

from spellcaster.spellcaster import run, train, Env


class SpellcasterMode(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"
    DEBUG = "debug"
    STANDBY = "standby"


class SpellcasterManager:
    def __init__(self):
        self.spellcaster_process = None
        self.mode = SpellcasterMode.STANDBY

    def terminate(self):
        if self.spellcaster_process is not None and self.spellcaster_process.is_alive():
            self.spellcaster_process.terminate()
            self.spellcaster_process.join()
        self.mode = SpellcasterMode.STANDBY

    def run(self):
        self.terminate()
        self.mode = SpellcasterMode.INFERENCE
        kwargs = {"debug": False, "env": Env.SUBPROCESS}
        self.spellcaster_process = Process(target=run, kwargs=kwargs)
        self.spellcaster_process.start()
    
    def train(self, spell_name: str):
        self.terminate()
        self.mode = SpellcasterMode.TRAINING
        kwargs = {"spell_name": spell_name, "env": Env.SUBPROCESS}
        self.spellcaster_process = Process(target=train, kwargs=kwargs)
        self.spellcaster_process.start()

    def debug(self):
        self.terminate()
        self.mode = SpellcasterMode.DEBUG
        kwargs = {"debug": True, "env": Env.SUBPROCESS}
        self.spellcaster_process = Process(target=run, kwargs=kwargs)
        self.spellcaster_process.start()
            
    def __del__(self):
        self.terminate()
    