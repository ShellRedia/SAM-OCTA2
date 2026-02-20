import numpy as np
from scipy import ndimage
import cv2, random
from collections import *
from itertools import *
from functools import *

class PromptGeneration:
    def __init__(self, random_seed=0, neg_range=(3, 20)):
        if random_seed:  
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.neg_range = neg_range

        self.min_area = 20

    def get_labelmap(self, label):
        structure = ndimage.generate_binary_structure(2, 2)
        labelmaps, connected_num = ndimage.label(label, structure=structure)

        label = np.zeros_like(labelmaps)
        for i in range(1, 1+connected_num):
            if np.sum(labelmaps==i) >= self.min_area: label += np.where(labelmaps==i, 255, 0)

        structure = ndimage.generate_binary_structure(2, 2)
        labelmaps, connected_num = ndimage.label(label, structure=structure)

        return labelmaps, connected_num
    
    def search_negative_region_numpy(self, labelmap):
        inner_range, outer_range = self.neg_range
        def search(neg_range):
            kernel = np.ones((neg_range * 2 + 1, neg_range * 2 + 1), np.uint8)
            negative_region = cv2.dilate(labelmap, kernel, iterations=1)
            mx = labelmap.max() + 1
            labelmap_r = (mx - labelmap) * np.minimum(1, labelmap)
            r = cv2.dilate(labelmap_r, kernel, iterations=1)
            negative_region_r = (r.astype(np.int32) - mx) * np.minimum(1, r)
            diff = negative_region.astype(np.int32) + negative_region_r
            overlap = np.minimum(1, np.abs(diff).astype(np.uint8))
            return negative_region - overlap - labelmap
        return search(outer_range) - search(inner_range) 

    def get_prompt_points(self, label_mask, ppp, ppn):
        label_mask_cp = np.copy(label_mask)
        label_mask_cp[label_mask_cp >= 1] = 1
        labelmaps, connected_num = self.get_labelmap(label_mask_cp)

        coord_positive, coord_negative = [], []

        connected_components = list(range(1, connected_num+1))
        random.shuffle(connected_components)

        for i in connected_components:
            cc = np.copy(labelmaps)
            cc[cc!=i] = 0
            cc[cc==i] = 1
            if ppp:
                coord_positive.append(random.choice([[y, x] for x, y in np.argwhere(cc == 1)]))
                ppp -= 1
        
        random.shuffle(connected_components)
        for i in connected_components:
            cc = np.copy(labelmaps)
            cc[cc!=i] = 0
            cc[cc==i] = 1
            negative_region = self.search_negative_region_numpy(cc.astype(np.uint8))
            negative_region = negative_region * (1 - label_mask_cp)
            if ppn:
                coord_negative.append(random.choice([[y, x] for x, y in np.argwhere(negative_region == 1)]))
                ppn -= 1

        negative_region = self.search_negative_region_numpy(label_mask_cp)

        if ppp: coord_positive += random.sample([[y, x] for x, y in np.argwhere(label_mask_cp == 1)], ppp)
        if ppn: coord_negative += random.sample([[y, x] for x, y in np.argwhere(negative_region == 1)], ppn)

        return coord_positive, coord_negative
    
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


