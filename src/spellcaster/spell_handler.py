from abc import ABC, abstractmethod


class SpellHandler(ABC):
    @abstractmethod
    def handle_spell(self, wand_path):
        pass


class InferenceSpellHandler(SpellHandler):
    def __init__(self, debug=False):
        self.debug = debug
        
    def handle_spell(self, wand_path):
        print("spell detected")


class TrainingSpellHandler(SpellHandler):
    def handle_spell(self, wand_path):
        print("spell detected")