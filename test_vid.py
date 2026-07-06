import tempfile
import os
import cv2

# Create dummy video bytes
class DummyUpload:
    def __init__(self):
        self.name = "myvideo.mp4"
        with open("dummy.mp4", "rb") as f:
            self.data = f.read()
    def read(self):
        return self.data

video_file = DummyUpload()

with tempfile.NamedTemporaryFile(delete=False, suffix="." + video_file.name.split(".")[-1]) as tmp:
    tmp.write(video_file.read())
    tmp_path = tmp.name

cap = cv2.VideoCapture(tmp_path)
print("Opened:", cap.isOpened())
cap.release()
os.unlink(tmp_path)
