import numpy as np
from scipy import ndimage
import cv2, random
from collections import *
from itertools import *
from functools import *

class PromptGeneration:
    def __init__(self, random_seed=0):
        if random_seed:  
            self.rng = random.Random(random_seed)
        else:
            self.rng = random.Random()

    def get_prompt_points(self, label_mask, point_num):
        prompt_points = []
        label_mask = np.where(label_mask>0, 1, 0)
        components, num_comp = ndimage.label(label_mask, np.ones((3, 3)))
        for i in range(1, num_comp+1):
            coords = [[y, x] for x,y in np.argwhere(components == i)]
            prompt_points += self.rng.sample(coords, 10)[:point_num]
        return prompt_points
    
# if __name__=="__main__":
#     pg = PromptGeneration()

#     label_mask = cv2.imread("test_mask.png", cv2.IMREAD_GRAYSCALE)
    
#     label_mask_3ch = cv2.imread("test_mask.png", cv2.IMREAD_COLOR)

#     coord_positive, coord_negative = pg.get_prompt_points(label_mask, 5, 3)

#     for x, y in coord_positive:
#         cv2.circle(label_mask_3ch, (x, y), 4, (0, 255, 0), -1)
    
#     for x, y in coord_negative:
#         cv2.circle(label_mask_3ch, (x, y), 4, (0, 0, 255), -1)
    
#     cv2.imwrite("test_mask_points.png", label_mask_3ch)


