import cv2
import numpy as np
from utils import draw_ai_visuals, generate_ai_heatmap

img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), -1)

bboxes = [[200, 200, 800, 800]] # ymin, xmin, ymax, xmax in 0-1000

boxed = draw_ai_visuals(img, bboxes)
heat = generate_ai_heatmap(img, bboxes)

print("Boxed shape:", boxed.shape)
print("Heat shape:", heat.shape)
