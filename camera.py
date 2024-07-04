import cv2

class Camera:
    def __init__(self):
        print("Starting video capture")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Couldn't start video capture")
        
    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Video capture ended abruptly")
            
            cv2.flip(frame,1)
            yield frame
    
    def __del__(self):
        print("Stopping video capture")
        self.cap.release()
