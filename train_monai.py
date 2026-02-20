import os
import time
import torch
import torch.nn as nn
from monai.networks.nets import *
from tqdm import tqdm
from options import parse_args
from collections import defaultdict
from loss_functions import clDiceLoss
from torch.utils.data import DataLoader

from octa_datasets import OCTA_Dataset_monai
from display import DisplaySequence
from metrics import MetricsStatistics

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

args = parse_args()

class MonaiTraining:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        self.result_dir = "/".join(["results", time_str, ""])
        self.result_dir += "_".join(map(str, [args.model_name, args.data_type, args.label_type, args.dataset]))
        os.makedirs(self.result_dir, exist_ok=True)

        self.frame_length = 6

        self.model = self.build_segmetation_model()

        if args.pretrained_weight_path:
            self.model.load_state_dict(torch.load(args.pretrained_weight_path))

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]        
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = clDiceLoss()

        init_dataset = lambda subset, frame_length: OCTA_Dataset_monai(
            dataset_name=args.dataset, 
            label_type=args.label_type, 
            data_type=args.data_type,
            subset=subset, 
            frame_length=frame_length
        )

        dataset_train = init_dataset("train", self.frame_length)
        dataset_eval_lst = [(init_dataset("val", self.frame_length), "val"), (init_dataset("test", self.frame_length), "test")]

        self.train_loader = DataLoader(dataset_train, batch_size=args.batch_size)
        
        self.eval_loader_lst = []
        for dataset, eval_info in dataset_eval_lst:
            self.eval_loader_lst.append([DataLoader(dataset, batch_size=1), eval_info])
        
        self.metrics_statistics = MetricsStatistics(save_dir=self.result_dir)
        self.ds = DisplaySequence()


    def build_segmetation_model(self):
        input_chs, output_chs = self.frame_length, self.frame_length
        if args.data_type == "single":
            input_chs, output_chs = 3, 1
        
        if args.model_name == "UNet":
            model = UNet(in_channels=3, out_channels=1, spatial_dims=2, channels=[512, 512, 1024, 2048, 2048], strides=[2, 2, 2, 2], kernel_size=9)
            return ModifiedModel(model).to(self.device)
        elif args.model_name == "SegResNet":
            model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2, init_filters=32, blocks_down=[2]*8,blocks_up=[2]*7)
            return ModifiedModel(model).to(self.device)
        elif args.model_name == "AttentionUnet":
            model = AttentionUnet(in_channels=3, out_channels=1, spatial_dims=2, channels=(128, 256, 512, 1024), strides=(2, 2, 2))
            return ModifiedModel(model).to(self.device)
        elif args.model_name == "SwinUNETR":
            model = SwinUNETR(in_channels=input_chs, out_channels=output_chs, feature_size=48, spatial_dims=2, use_v2=True)
            return ModifiedModel(model).to(self.device)
        elif args.model_name == "DiNTS":
            dints_space = TopologyInstance(spatial_dims=2, num_blocks=12, device="cuda")
            model = DiNTS(dints_space=dints_space, in_channels=input_chs, num_classes=output_chs, spatial_dims=2)
            return ModifiedModel(model).to(self.device)
        
    def train(self):
        progress_bar = tqdm(range(args.epochs))

        for epoch in range(args.epochs+1):
            if epoch:
                logs = self.train_epoch()
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)

            # self.evaluate(epoch, self.train_loader, "train")
            if epoch + 5 >= args.epochs or epoch % 10 == 0:
                for dataloader, eval_info in self.eval_loader_lst:
                    self.evaluate(epoch, dataloader, eval_info)

                # save_result
                self.metrics_statistics.record_result(epoch)
                torch.save(self.model.state_dict(), '{}/{:0>4}.pth'.format(self.result_dir,epoch))

    def train_epoch(self):
        self.model.train()
        logs = defaultdict(list)
        self.optimizer.zero_grad()
        
        for images, masks, _ in self.train_loader:
            images, masks = map(self.to_cuda, [images, masks])
            
            preds = self.model(images)
            loss = self.criterion(preds, masks)
            loss.backward() 
        
            self.optimizer.step()
            self.optimizer.zero_grad()

            logs["loss"].append(loss.item())

        return {k : round(sum(v) / len(v), 4) for k, v in logs.items()}

    @torch.no_grad()
    def evaluate(self, epoch, eval_loader, subset="val"):
        self.model.eval()

        to_cpu = lambda x: x[0].int()

        for images, masks, sample_ids in tqdm(eval_loader, "metrics calculation", leave=False):
            images, masks = map(self.to_cuda, [images, masks])
            preds = self.model(images)

            preds = torch.where(preds > 0.7, 1.0, 0.0)

            images, masks, preds = images.cpu(), masks.cpu(), preds.cpu()
            mask, pred = to_cpu(masks), to_cpu(preds)

            metric_name = "{}-{}".format(args.label_type, subset)

            if mask.shape == pred.shape:
                for i in range(mask.shape[0]):
                    self.metrics_statistics.cal_epoch_metric(args.metrics, metric_name, mask[i:i+1], pred[i:i+1], epoch>0)        
            else:
                self.metrics_statistics.cal_epoch_metric(args.metrics, metric_name, mask, pred, epoch>0)
        
            sample_save_dir = "{}/{:0>4}/{}".format(self.result_dir, epoch, subset)
            os.makedirs(sample_save_dir, exist_ok=True)

            self.ds.display_monai_prediction(images[0], masks[0], preds[0], sample_ids[0], sample_save_dir)

if __name__=="__main__":
    monai_trainer = MonaiTraining()
    monai_trainer.train()