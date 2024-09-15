import cv2
import os
import numpy as np
from collections import Counter
from skimage import morphology

alpha = 0.5
overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_color = lambda x, color: (to_3ch(x) * color).astype(dtype=np.uint8)

def mark_rv_objects(sample_name):
    layer_root_dir = "datasets/OCTA500/RV_LayerSequence"
    colored_rv_dir = "datasets/OCTA500/Colored_RV"
    to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

    colors = [tuple(x) for x in np.load("color_list.npy")]
    colors_dct = dict(zip(colors, range(len(colors))))
    colored_mask_rv = cv2.imread("{}/{}.png".format(colored_rv_dir, sample_name), cv2.IMREAD_COLOR)

    sample_dir = "{}/{}".format(layer_root_dir, sample_name)
    ref_mask_dir = "{}/ref_mask".format(sample_dir)
    os.makedirs(ref_mask_dir, exist_ok=True)

    max_color_diff, min_component_lmt = 10, 100

    area_cnts = Counter()

    for layer_file in [x for x in sorted(os.listdir(sample_dir)) if ".png" in x]:
        layer_name = layer_file[:-4]
        layer_dir = "{}/{}".format(sample_dir, layer_name)
        
        os.makedirs(layer_dir, exist_ok=True)
        
        layer_image = cv2.imread("{}/{}".format(sample_dir, layer_file), cv2.IMREAD_COLOR)
        _, w, _ = layer_image.shape
        layer_mask = np.expand_dims(layer_image[:,w//2:3*w//4,1] - layer_image[:,w//2:3*w//4,0], axis=-1) 
        layer_mask[layer_mask >= 1] = 1

        cleaned_mask = morphology.remove_small_objects(layer_mask.astype(bool), min_size=min_component_lmt)
        rv_mask = colored_mask_rv * cleaned_mask

        cv2.imwrite("{}/{}.png".format(ref_mask_dir, layer_name), np.concatenate([to_3ch(layer_mask[:,:,0]) * 255, rv_mask], axis=1))

        for pixel_color, color_id in list(colors_dct.items())[:25]:
            diff_map = np.abs(rv_mask - pixel_color)
            diff_map = np.sum(diff_map, axis=2)
            object_mask = np.where(diff_map < max_color_diff, 1, 0).astype(np.uint8)

            if np.sum(object_mask) >= min_component_lmt:
                area_cnts[color_id] += np.sum(object_mask)
                cv2.imwrite("{}/{:0>2}.png".format(layer_dir, color_id), object_mask * 255)
    
    
    selected_rvs = set(area_cnts.keys())

    for layer_dir in [x for x in sorted(os.listdir(sample_dir)) if ".png" not in x]:
        object_lst = []
        if layer_dir == "ref_mask": continue
        object_dir = "{}/{}".format(sample_dir, layer_dir)
        for mask_file in os.listdir(object_dir):
            object_id = int(mask_file[:-4])
            if object_id not in selected_rvs: os.remove("{}/{}".format(object_dir, mask_file))
            else: object_lst.append(object_id)
        
        np.save("{}/objects.npy".format(object_dir), np.array(object_lst))

def extract_modified_rv_mask(sample_name):
    image_dir = "datasets/OCTA500/RV_LayerSequence/{}".format(sample_name)
    image_files = sorted([x for x in os.listdir(image_dir) if ".png" in x])
    erased = cv2.imread("erase_region.png", cv2.IMREAD_GRAYSCALE)
    erased_lst = []

    rv_mask_file = "datasets/OCTA500/3M/GT_LargeVessel/{}.bmp".format(sample_name)
    rv = cv2.imread(rv_mask_file, cv2.IMREAD_COLOR)
    rv = cv2.resize(rv, (512, 512))
    rv[rv>=1] = 1

    h, w = erased.shape
    for i in range(10):
        erased_image = erased[:, i*(20+h):i*(20+h)+h]
        erased_image[erased_image>=1] = 1
        erased_lst.append(erased_image * 255)

    for i, image_file in enumerate(image_files):
        image = cv2.imread("{}/{}".format(image_dir, image_file))
        _, w, _ = image.shape
        e = erased_lst[i]
        a, b, c, d = image[:,:w//4], image[:,w//4:w//2], image[:,w//2:-w//4], image[:,-w//4:]
        a -= to_3ch(e)
        a[a<255] = 0
        b = overlay((a * (0, 1, 0)).astype(np.uint8), (d * 2).astype(np.uint8))
        rv_mask = rv * a
        rv_mask = morphology.remove_small_objects(rv_mask.astype(bool), min_size=50)
        c = overlay((rv_mask * (0, 255, 0)).astype(np.uint8), (d * 2).astype(np.uint8))
        cv2.imwrite("{}/{}".format(image_dir, image_file), np.concatenate([a, b, c, d], axis=1))

