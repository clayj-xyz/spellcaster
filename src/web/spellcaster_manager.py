from multiprocessing import Process

from spellcaster.spellcaster import build_spellcaster, SpellcasterMode, RunMode


class SpellcasterManager:
    def __init__(self):
        self.spellcaster_process = None
        self.mode = SpellcasterMode.STANDBY

    def terminate(self):
        if self.spellcaster_process is not None and self.spellcaster_process.is_alive():
            self.spellcaster_process.terminate()
            self.spellcaster_process.join()
        self.mode = SpellcasterMode.STANDBY
            
    def run_spellcaster(self):
        spellcaster = build_spellcaster(
            run_mode=RunMode.SUBPROCESS,
            spellcaster_mode=self.mode
        )
        spellcaster.run()

    def change_mode(self, new_mode: SpellcasterMode):
        self.terminate()
        if new_mode != SpellcasterMode.STANDBY:
            self.mode = new_mode
            self.spellcaster_process = Process(target=self.run_spellcaster)
            self.spellcaster_process.start()
            
    def __del__(self):
        self.terminate()
    