import cv2
import numpy as np

# create a dummy video
out = cv2.VideoWriter("dummy.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (100, 100))
for i in range(50):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(frame, (50, 50), i, (255, 255, 255), -1)
    out.write(frame)
out.release()

from video_processing import process_video_frames
res = process_video_frames("dummy.mp4", 5)
print("Result frames:", len(res))
