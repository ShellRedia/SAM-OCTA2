import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imgaug.augmenters as iaa

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from monai.networks.nets import *
from torch.utils.data import Dataset

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x
    
class LayerVesselRegionDataset(Dataset):
    def __init__(
            self, 
            data_dir="datasets/Sparse/OCTA-500/sam2_region",
        ):
        self.data_dir = data_dir
        self.sample_files = sorted(os.listdir(data_dir))

        self.seq = iaa.Sequential([
            iaa.Fliplr(0.1), # horizontal flips
            iaa.Flipud(0.1), # vertical flips
            iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1))),
            iaa.Sometimes(0.1, iaa.LinearContrast((0.75, 1.5))),
            iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
            iaa.Sometimes(0.1, iaa.Rotate(rotate=(-10, 10), mode='constant')),
            iaa.Sometimes(0.1, iaa.Sharpen((0.0, 0.5))),
            iaa.Sometimes(0.1, iaa.ElasticTransformation(sigma=15)),
            iaa.Sometimes(0.1, iaa.ImpulseNoise(p=0.05)),
            iaa.Sometimes(0.1, iaa.Dropout([0.05, 0.2])),
        ], random_order=True) # apply augmenters in random order

    def convert_image_to_tensor(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        else:
            if image.shape[2] < 10:
                image = np.transpose(image, (2, 0, 1))

        if image.max() > 1: image = image / 255
        return torch.tensor(image)
    
    def data_augmentation(self, image, mask):
        to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
        is_image_2ch, is_mask_2ch = bool(len(image.shape) == 2), bool(len(mask.shape) == 2)
        if is_image_2ch: image = to_3ch(image)
        if is_mask_2ch: mask = to_3ch(mask)
        images, masks = np.expand_dims(image, axis=0), np.expand_dims(mask, axis=0)
        images_aug, masks_aug = self.seq(images=images, segmentation_maps=masks)
        images_aug, masks_aug = images_aug[0], masks_aug[0]
        if is_image_2ch: images_aug = images_aug[:,:,0]
        if is_mask_2ch: masks_aug = masks_aug[:,:,0]
        return images_aug, masks_aug

        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, index):
        sample = cv2.imread("{}/{}".format(self.data_dir, self.sample_files[index]), cv2.IMREAD_GRAYSCALE)

        h = sample.shape[0]
        image, mask = sample[:,:h], sample[:,h:]
        image, mask = self.data_augmentation(image, mask)
        image, mask = map(self.convert_image_to_tensor, (image, mask))

        return image, mask
    
    
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss
    
class TrainingManager:
    def __init__(self):
        self.epochs = 100
  
        self.cpt_dir = "pretrained_weights"
        os.makedirs(self.cpt_dir, exist_ok=True)

        dataset_train = LayerVesselRegionDataset()

        self.train_loader  = DataLoader(dataset_train, batch_size=4)
        self.loss_func = DiceLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        dints_space = TopologyInstance(spatial_dims=2, num_blocks=6, device="cuda")
        model = DiNTS(dints_space=dints_space, in_channels=1, num_classes=1, spatial_dims=2)
        self.model = ModifiedModel(model).to(self.device)

        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=1e-4)

    def train(self):
        progress_bar = tqdm(range(self.epochs))
        self.model.train()
        for epoch in range(self.epochs):
            logs = defaultdict(list)
            for images, labels in tqdm(self.train_loader, "train batches", leave=False):
                images, labels = map(self.to_cuda, [images, labels])
                self.optimizer.zero_grad()
                preds = self.model(images)
                loss = self.loss_func(preds, labels)
                loss.backward()
                self.optimizer.step()

                logs["loss"].append(loss.item())

            logs = {k : round(sum(v) / len(v), 4) for k, v in logs.items()}

            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            torch.save(self.model.state_dict(), '{}/dints_region.pth'.format(self.cpt_dir))

if __name__=="__main__":
    trainer = TrainingManager()
    trainer.train()