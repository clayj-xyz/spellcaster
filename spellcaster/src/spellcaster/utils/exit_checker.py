import signal
from abc import ABC, abstractmethod

import cv2 as cv2


class ExitChecker(ABC):
    @abstractmethod
    def should_exit(self):
        pass


class SigTermExitChecker(ExitChecker):
    def __init__(self):
        self.exit = False
        signal.signal(signal.SIGINT, self.set_exit)
        signal.signal(signal.SIGTERM, self.set_exit)

    def set_exit(self, signum, frame):
        self.exit = True

    def should_exit(self):
        return self.exit
    

class CV2ExitChecker(ExitChecker):
    def should_exit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')