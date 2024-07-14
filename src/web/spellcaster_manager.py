from multiprocessing import Process

from spellcaster.wand_tracker import serve_wand_tracker


class SpellcasterManager:
    def __init__(self):
        self.spellcaster_process = None
        self.mode = "standby"

    def change_mode(self, new_mode):
        if new_mode == self.mode:
            return
        
        if new_mode == "standby":
            self.spellcaster_process.terminate()
        elif new_mode == "active":
            self.spellcaster_process = Process(target=serve_wand_tracker)
            self.spellcaster_process.start()
        elif new_mode == "training":
            print("training")
        else:
            raise ValueError("invalid mode")
        
        self.mode = new_mode

    def __del__(self):
        if self.spellcaster_process is not None:
            self.spellcaster_process.terminate()
    