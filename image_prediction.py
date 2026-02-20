import os
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# hydra.core.global_hydra.GlobalHydra.instance().clear()

alpha = 0.5
overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, checkpoint)

predictor = SAM2ImagePredictor(sam2_model)

image_path = "images/truck.jpg"
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

point_coords = np.array([[500, 375]])
point_labels = np.array([1])

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords, point_labels)

# pred = overlay()
    
image = cv2.imread(image_path)
pred = overlay(image, to_yellow(masks[0] * 255))

pred_dir = "predictions"
os.makedirs(pred_dir, exist_ok=True)
cv2.imwrite("{}/pred.png".format(pred_dir), pred)