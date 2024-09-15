import torch
import torch.nn as nn
import time

import os
import cv2
import numpy as np
from tqdm import tqdm

from skimage import measure, morphology, filters
from scipy.ndimage import binary_fill_holes, gaussian_filter

from monai.networks.nets import *
from octa_datasets import OCTA_Dataset_Layer_Sparse_Annotation_Prediction
from torch.utils.data import DataLoader




class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

    
class PredictManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])

        alpha = 0.5
        self.overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
        self.to_color = lambda x, color: (self.to_3ch(x) * color).astype(dtype=np.uint8)
    
        self.model = SwinUNETR(img_size=(512,512), in_channels=1, out_channels=1, feature_size=72, spatial_dims=2)
        self.model = ModifiedModel(self.model).to(self.device)

        weight_path = "checkpoints/your_checkpoint_path.pth"
        weight_path = r"result\SparseAnnotation_RV\2024-09-14-14-52-39\rv_predictor_40.pth"
        
        self.model.load_state_dict(torch.load(weight_path))

        self.dataset= OCTA_Dataset_Layer_Sparse_Annotation_Prediction()

        self.dataloader = DataLoader(self.dataset, batch_size=1)
    
    def predict(self):
        to_cpu = lambda x:x[0][0].cpu().detach().int()
        
        save_dir = "result/SparseAnnotation_RV_pred/{}".format(self.time_str)

        progress_bar = tqdm(range(len(self.dataset)))

        for samples, (sample_name, layer_name) in self.dataloader:
            sample_name, layer_name = sample_name.item(), layer_name[0][:-4]
            preds = self.model(samples.to(self.device))
            preds = torch.gt(preds, 0.8).int()

            fov = "3M" if sample_name >= 10301 else "6M"

            rv_gt_dir = "datasets/OCTA500/{}/GT_LargeVessel".format(fov)
            rv_mask = cv2.imread("{}/{}.bmp".format(rv_gt_dir, sample_name), cv2.IMREAD_GRAYSCALE)
            rv_mask = cv2.resize(rv_mask, (512, 512)) / 255

            pred_mask = to_cpu(preds).numpy().astype(np.uint8)

            pred_mask = self.smooth_mask(pred_mask)
            # pred_mask *= 255
            
            save_sample_dir = "{}/{}".format(save_dir, sample_name)
            os.makedirs(save_sample_dir, exist_ok=True)

            sample_image = self.to_3ch(to_cpu(samples * 255).numpy())

            overlay_mask = self.overlay(sample_image * 2, self.to_color(pred_mask, (0,1,1)))
            rv_mask = self.overlay(sample_image * 2, self.to_color(pred_mask * rv_mask, (0,1,0)))

            pred_mask = self.to_3ch(pred_mask)

            layer_image = np.concatenate([pred_mask, overlay_mask, rv_mask, sample_image], axis=1)

            cv2.imwrite("{}/{}.png".format(save_sample_dir, layer_name), layer_image)

            progress_bar.update(1)

    def smooth_mask(self, mask):
        # remove small components and fulfill holes
        labeled_image = measure.label(mask)
        cleaned_image = morphology.remove_small_objects(labeled_image, min_size=500)
        filled_image = binary_fill_holes(cleaned_image).astype(np.uint8)

        smoothed_image_gauss = gaussian_filter(filled_image.astype(float), sigma=5)
        smoothed_image_gauss = smoothed_image_gauss > filters.threshold_otsu(smoothed_image_gauss)
        
        return smoothed_image_gauss * 255
    


if __name__=="__main__":
    pm = PredictManager()
    pm.predict()