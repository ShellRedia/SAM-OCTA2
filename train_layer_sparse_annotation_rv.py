import torch
import torch.nn as nn
import torch.optim as optim
import time

import os
import random
import numpy as np
from tqdm import tqdm

from monai.networks.nets import *
from octa_datasets import OCTA_Dataset_Layer_Sparse_Annotation_Training
from torch.utils.data import DataLoader

import pandas as pd
from collections import defaultdict


class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

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
    
class TrainManager:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = 500
        self.lr = 1e-4
        self.epsilon = 1e-8

        self.time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])

        self.save_interval = 10

        self.model = SwinUNETR(img_size=(512,512), in_channels=1, out_channels=1, feature_size=72, spatial_dims=2)

        self.model = ModifiedModel(self.model).to(device)
    
        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = self.epochs // 5
        lr_lambda = lambda x: max(1e-4, self.lr * x / epoch_p if x <= epoch_p else self.lr * 0.97 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        dataset_train = OCTA_Dataset_Layer_Sparse_Annotation_Training()

        self.train_loader = DataLoader(dataset_train, batch_size=8)

        self.inputs_process = lambda x : x.to(device)
        self.loss_func = DiceLoss()

    def cal_jaccard_index(self, pred, label):
        intersection = (pred & label).sum().item()
        union = (pred | label).sum().item()
        jaccard_index = intersection / (union + self.epsilon)
        return jaccard_index

    def cal_dice(self, pred, label):
        intersection = (pred & label).sum().item()
        union = pred.sum().item() + label.sum().item()
        dice = 2 * intersection / (union + self.epsilon)
        return dice
        
    
    def train(self):
        metrics = defaultdict(list)

        to_cpu = lambda x:x[0][0].cpu().detach().int()
                    
        progress_bar = tqdm(range(self.epochs))
        
        save_dir = "result/RV_Layer/{}".format(self.time_str)
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), '{}/rv_predictor_{}.pth'.format(save_dir, 0))

        # training loop:
        for epoch in range(1, self.epochs+1):
            dice_lst, jac_lst = [], []
            for samples, masks in self.train_loader:
                samples, masks = map(self.inputs_process, (samples, masks))
                self.optimizer.zero_grad()
                preds = self.model(samples)
                self.loss_func(preds, masks).backward()
                self.optimizer.step()
                preds = torch.gt(preds, 0.8).int()
                masks = torch.gt(masks, 0.8).int()
                masks, preds = to_cpu(masks), to_cpu(preds)
                dice_lst.append(self.cal_dice(masks, preds))
                jac_lst.append(self.cal_jaccard_index(masks, preds))

            self.scheduler.step()

            metrics["epoch"].append(epoch)
            dice_mean, jac_mean = round(sum(dice_lst) / len(dice_lst), 4), round(sum(jac_lst) / len(jac_lst), 4)
            metrics["dice"].append(dice_mean)
            metrics["jac"].append(jac_mean)



            pd.DataFrame(metrics).to_excel("{}/metrics.xlsx".format(save_dir))

            progress_bar.update(1)
            logs = {"dice": dice_mean,"jac": jac_mean}
            progress_bar.set_postfix(**logs)

            if epoch % self.save_interval == 0:
                torch.save(self.model.state_dict(), '{}/rv_predictor_{}.pth'.format(save_dir, epoch))

if __name__=="__main__":
    tm = TrainManager()
    tm.train()