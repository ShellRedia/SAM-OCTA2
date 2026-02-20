import os
import cv2
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from options import parse_args
from sam2_train.build_sam import build_sam2_video_predictor
from peft import LoraConfig, get_peft_model

from collections import defaultdict

from loss_functions import DiceLoss
from octa_datasets import OCTA_Dataset_SAM2_Training, OCTA_Dataset_SAM2_Projection
from display import DisplaySequence
from metrics import MetricsStatistics

args = parse_args()

class VideoFineTune:
    def __init__(self, fov):
        config_file, ckpt_path = {
            "large":("sam2_hiera_l.yaml", "./sam2_weights/sam2_hiera_large.pt"),
            "base_plus":("sam2_hiera_b+.yaml", "./sam2_weights/sam2_hiera_base_plus.pt"),
            "small":("sam2_hiera_s.yaml", "./sam2_weights/sam2_hiera_small.pt"),
            "tiny":("sam2_hiera_t.yaml", "./sam2_weights/sam2_hiera_tiny.pt"),
        }[args.model_type]

        self.device = torch.device("cuda:0")

        self.time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        self.result_dir = "results/{}".format(self.time_str)

        os.makedirs(self.result_dir, exist_ok=True)

        self.metrics_statistics = MetricsStatistics(save_dir=self.result_dir)

        self.sam2 = build_sam2_video_predictor(config_file=config_file, ckpt_path=ckpt_path, mode=None, apply_postprocessing=False)
        
        for param in self.sam2.parameters(): param.requires_grad = False

        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
        
        self.sam2.image_encoder = get_peft_model(self.sam2.image_encoder, lora_config)

        self.lora_sam2 = self.sam2
        
        trainable_params = [param for param in self.lora_sam2.parameters() if param.requires_grad]

        self.optimizer = torch.optim.Adam(trainable_params, lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.criterion = DiceLoss()

        if args.task_type == "layer_sequence":
            dataset_train = OCTA_Dataset_SAM2_Training(validation_mode=False, fov=fov, label_type=args.label_type)
            dataset_val = OCTA_Dataset_SAM2_Training(validation_mode=True, fov=fov, label_type=args.label_type)
        
        else:
            dataset_train = OCTA_Dataset_SAM2_Projection(subset_name="Training", fov=fov, label_type=args.label_type)
            dataset_val = OCTA_Dataset_SAM2_Projection(subset_name="Validation", fov=fov, label_type=args.label_type)
            dataset_test = OCTA_Dataset_SAM2_Projection(subset_name="Test", fov=fov, label_type=args.label_type)


        self.train_loader = DataLoader(dataset_train, batch_size=1)
        self.val_loader = DataLoader(dataset_val, batch_size=1)

        self.ds = DisplaySequence()
    
    def train(self):
        self.lora_sam2.train()

        progress_bar = tqdm(range(args.epochs))
        self.evaluate(0)

        for epoch in range(args.epochs):
            logs = self.train_epoch(epoch)
            self.evaluate(epoch+1)
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

        self.metrics_statistics.close()

    def train_epoch(self, epoch):
        rnt_g = defaultdict(list)

        for batch in self.train_loader:
            image_seq = batch["images"]
            image_seq = image_seq.squeeze(0).to(dtype=torch.float32, device=self.device)

            video_length = len(image_seq)

            mask_seq = batch["masks"]
            prompt_points = batch["prompts"]

            train_state = self.lora_sam2.train_init_state(imgs_tensor=image_seq)
            self.lora_sam2.reset_state(train_state)

            for frame_idx in prompt_points:
                for obj_id in prompt_points[frame_idx]:
                    points = prompt_points[frame_idx][obj_id]["points"]
                    labels = prompt_points[frame_idx][obj_id]["labels"]

                    _, out_obj_ids, out_mask_logits = self.lora_sam2.train_add_new_points(
                        inference_state=train_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points.to(device=self.device),
                        labels=labels.to(device=self.device),
                        clear_old_points=False,
                    )

            video_segments = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in self.lora_sam2.train_propagate_in_video(train_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}

            loss = 0

            for frame_idx in range(video_length):
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    pred = torch.sigmoid(out_mask[0].unsqueeze(0))
                    mask = mask_seq[frame_idx][out_obj_id].to(dtype=torch.float32, device=self.device)

                loss += self.criterion(mask, pred)

            rnt_g["loss"].append(loss.item())
            loss.backward(retain_graph=True)
            self.optimizer.step()

        if (epoch+1) % 5 == 0: torch.save(self.lora_sam2.state_dict(), '{}/{}_{:0>3}.pth'.format(self.result_dir, args.label_type, epoch+1))
        return {k : round(sum(v) / len(v), 4) for k, v in rnt_g.items()}
    
    def evaluate(self, epoch):
        sample_name_p = ""
        batch_i = 0
        for batch in self.val_loader:
            sample_name = batch["sample_name"][0]
            if sample_name == sample_name_p:
                batch_i += 1
            else:
                sample_name_p = sample_name
                batch_i = 0
            image_seq = batch["images"]
            video_length = image_seq.shape[1]
            image_seq = image_seq.squeeze(0).to(device=self.device)

            mask_seq = batch["masks"]
            prompt_points = batch["prompts"]

            train_state = self.lora_sam2.train_init_state(imgs_tensor=image_seq)
            self.lora_sam2.reset_state(train_state)

            for frame_idx in prompt_points:
                for obj_id in prompt_points[frame_idx]:
                    points = prompt_points[frame_idx][obj_id]["points"].squeeze(0).to(device=self.device)
                    labels = prompt_points[frame_idx][obj_id]["labels"].squeeze(0).to(device=self.device)

                    _, out_obj_ids, out_mask_logits = self.lora_sam2.train_add_new_points(
                        inference_state=train_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        clear_old_points=False,
                    )

            video_segments = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in self.lora_sam2.train_propagate_in_video(train_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] > 0.0 for i, out_obj_id in enumerate(out_obj_ids)}
            
            pred_seq = []
            for frame_idx in range(video_length):
                pred_dct = {}
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    pred_dct[out_obj_id] = out_mask[0].cpu().numpy()
                    mask_gt = mask_seq[frame_idx][out_obj_id]
                    if args.task_type == "projection":
                        if frame_idx == 1:
                            self.metrics_statistics.cal_epoch_metric(args.metrics, "{}-{}".format(args.label_type, "val"), mask_gt.int(), out_mask[0].cpu().int())
                    else:
                        self.metrics_statistics.cal_epoch_metric(args.metrics, "{}-{}".format(args.label_type, "val"), mask_gt.int(), out_mask[0].cpu().int())
                pred_seq.append(pred_dct)

            sample_save_dir = "{}/{}_{}/{:0>4}".format(self.result_dir, args.label_type, args.task_type, epoch)
            if args.task_type == "layer_sequence": image_seq = image_seq * 2
            self.ds.display_a_batch_prediction(image_seq , mask_seq, pred_seq, prompt_points, sample_save_dir, "{}_{}".format(sample_name, batch_i))
        
        self.metrics_statistics.record_result(epoch)
    
    def print_trainable_parameters(self, model):
        trainable_params, all_param = 0, 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


if __name__=="__main__":
    video_fine_tune = VideoFineTune(fov="6M")
    video_fine_tune.train()