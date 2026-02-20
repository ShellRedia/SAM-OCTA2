import os
import cv2
import numpy as np

sample_id = 10441
image_dir = "datasets/OCTA500/RV_LayerSequence/{}".format(sample_id)

image_files = sorted([x for x in os.listdir(image_dir) if ".png" in x])


image_lst = []
for i, image_file in enumerate(image_files):
    image = cv2.imread("{}/{}".format(image_dir, image_file))
    _, w, _ = image.shape

    padding = np.ones((w//4, 20, 3)) * (0, 0, 255)
    image_lst.append(image[:,w//4:w//2])
    image_lst.append(padding.astype(np.uint8))

merged = np.concatenate(image_lst, axis=1)

cv2.imwrite("merged.png", merged)