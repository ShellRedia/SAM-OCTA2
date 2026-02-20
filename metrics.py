from collections import *
from statistics import mean
from skimage.morphology import skeletonize
import pandas as pd
import numpy as np
import medpy.metric.binary as bm
from scipy import ndimage

class MetricsStatistics:
    def __init__(self, save_dir="./results/"):
        self.epsilon = 1e-6
        self.func_dct = {"Dice": bm.dc, "Jaccard": bm.jc, "clDice": self.clDice, "Hausdorff": bm.hd, "HD95": bm.hd95}
        self.save_dir = save_dir
        self.metric_values = defaultdict(list)
        self.metric_epochs = defaultdict(list)

    def cal_epoch_metric(self, metrics, label_type, label, pred, is_rv_frag=True):
        label, pred = map(lambda x:x.numpy(), [label, pred])
        if is_rv_frag: pred = self.remove_fragments(pred)
        for x in metrics:
            if np.sum(label):
                metric_value = self.func_dct[x](label, pred) if np.sum(pred) else 0
                self.metric_values["{}-{}".format(x, label_type)].append(metric_value)

    def record_result(self, epoch):
        self.metric_epochs["epoch"].append(epoch)
        for k, v in self.metric_values.items():
            self.metric_epochs[k].append(str(round(mean(v),4)))
        pd.DataFrame(self.metric_epochs).to_excel("{}/metrics_statistics.xlsx".format(self.save_dir), index=False)
        self.metric_values.clear()
    
    def clDice(self, v_p, v_l):
        epsilon = 1e-8
        cl_score = lambda v, s: np.sum(v*s)/np.sum(s)
        
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))

        return 2*tprec*tsens/(tprec+tsens+epsilon)
    
    def remove_fragments(self, binary_img, fragment_area=50):
        binary_img = binary_img[0]
        binary_img = np.where(binary_img>0, 1, 0)
        labeled, num = ndimage.label(binary_img, structure=np.ones((3,3)))
        for i in range(1, num+1):
            area = np.sum(labeled == i)
            if area < fragment_area:
                binary_img[labeled == i] = 0
        
        return np.array([np.where(binary_img>0, 1, 0)])