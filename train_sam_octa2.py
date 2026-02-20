import os
import cv2
import time
from matplotlib import artist
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from itertools import product
from options import parse_args
from sam2_train.build_sam import build_sam2_video_predictor
from fine_tune_modules import SAM2_Adapter

from collections import defaultdict

from loss_functions import clDiceLoss, DiceLoss
from octa_datasets import OCTA_Dataset_SAM2_Sequence, OCTA_Dataset_SAM2_Single
from display import DisplaySequence
from metrics import MetricsStatistics

args = parse_args()

class VideoFineTune:
    def __init__(self):
        # training informations:
        self.device = torch.device("cuda:0")
        time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        self.result_dir = "/".join(["results", time_str])
        self.result_dir += "/" + "_".join(map(str, [args.model_type, args.data_type, args.is_local, args.label_type, args.dataset, args.adapter, args.rank]))
        os.makedirs(self.result_dir, exist_ok=True)
        
        # model loading
        ckpt_name = {"large":"l", "base_plus":"b+", "small":"s", "tiny":"t"}[args.model_type]
        config_file = "sam2_hiera_{}.yaml".format(ckpt_name)
        ckpt_path = "pretrained_weights/sam2_hiera_{}.pt".format(args.model_type)
        self.sam2 = build_sam2_video_predictor(config_file=config_file, ckpt_path=ckpt_path, mode=None, apply_postprocessing=False)

        self.load_adapter()

        if args.pretrained_weight_path:
            self.lora_sam2.load_state_dict(torch.load(args.pretrained_weight_path.format(self.result_dir)))

        # optimizer and loss function
        trainable_params = [param for param in self.lora_sam2.parameters() if param.requires_grad]        
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = clDiceLoss()


        if args.data_type == "sequence":
            init_dataset = lambda subset, frame_length, prompt_frames, prompt_num: OCTA_Dataset_SAM2_Sequence(
                dataset_name=args.dataset, 
                label_type=args.label_type, 
                subset=subset, 
                frame_length=frame_length,
                prompt_frames=prompt_frames,
                prompt_num=prompt_num,
                is_local=bool(args.is_local=="Local")
            )
            dataset_train = init_dataset("train", 2, 2, 2)
            dataset_eval_lst = []

            for fl, pf, pn in product(args.frame_length, args.prompt_frames, args.prompt_num):
                dataset_eval_lst.append([init_dataset("val", fl, pf, pn), "-".join(map(str, ["val", fl, pf, pn]))])
                dataset_eval_lst.append([init_dataset("test", fl, pf, pn), "-".join(map(str, ["test", fl, pf, pn]))])

        else:
            init_dataset = lambda subset: OCTA_Dataset_SAM2_Single(
                dataset_name=args.dataset, 
                label_type=args.label_type, 
                subset=subset)
            dataset_train = init_dataset("train")
            dataset_eval_lst = [[init_dataset("val"), "val"], [init_dataset("test"), "test"]]



        self.train_loader = DataLoader(dataset_train, batch_size=1)
        self.eval_loader_lst, self.test_loader_lst = [], []

        for dataset, eval_info in dataset_eval_lst:
            self.eval_loader_lst.append([DataLoader(dataset, batch_size=1), eval_info])

        # recording
        self.metrics_statistics = MetricsStatistics(save_dir=self.result_dir)
        self.ds = DisplaySequence()

    def load_adapter(self):
        for param in self.sam2.image_encoder.parameters(): param.requires_grad = False
        adapter = SAM2_Adapter()
        adapter.check_image_encoder_structure(self.sam2)
        self.lora_sam2 = adapter.attach_adapter(self.sam2, args.rank, args.adapter).cuda()
        self.lora_sam2.image_encoder.trunk.use_checkpoint = True
    
    def train(self):
        self.lora_sam2.train()
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
                torch.save(self.lora_sam2.state_dict(), '{}/{:0>4}.pth'.format(self.result_dir,epoch))

    def train_epoch(self):
        logs = defaultdict(list)
        accumulation_steps = args.batch_size
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            video_length, mask_seq, train_state = self.batch_to_model(batch)
    
            video_segments = {} 
            for out_frame_idx, out_obj_ids, out_mask_logits in self.lora_sam2.train_propagate_in_video(train_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}

            loss = 0
            for frame_idx in range(video_length):
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    pred = torch.sigmoid(out_mask[0].unsqueeze(0))
                    mask = mask_seq[frame_idx][out_obj_id].to(dtype=torch.float32, device=self.device)
                    loss += self.criterion(mask, pred)
            
            loss = loss / (video_length * accumulation_steps) 
            
            loss.backward() 
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.lora_sam2.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            logs["loss"].append(loss.item() * accumulation_steps)

        return {k : round(sum(v) / len(v), 4) for k, v in logs.items()}

    @torch.no_grad()
    def evaluate(self, epoch, eval_loader, subset="val"):
        for batch in eval_loader:
            mask_seq = batch["masks"]

            video_length, mask_seq, train_state = self.batch_to_model(batch)
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.lora_sam2.train_propagate_in_video(train_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] > 0.7 for i, out_obj_id in enumerate(out_obj_ids)}
            
            pred_seq = []
            for frame_idx in range(video_length):
                pred_dct = {}
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    pred_dct[out_obj_id] = out_mask[0].cpu().numpy()
                    mask_gt = mask_seq[frame_idx][out_obj_id]
                    metric_name = "{}-{}".format(args.label_type, subset)

                    self.metrics_statistics.cal_epoch_metric(args.metrics, metric_name, mask_gt.int(), out_mask.cpu().int())
                pred_seq.append(pred_dct)

            sample_save_dir = "{}/{:0>4}/{}".format(self.result_dir, epoch, subset)
            os.makedirs(sample_save_dir, exist_ok=True)
            if (args.dataset == "3M" or args.dataset == "6M") and args.data_type=="sequence":
                batch["images"] *= 2 # increase the brightness
            self.ds.display_a_batch_prediction(batch, pred_seq, sample_save_dir)

    def batch_to_model(self, batch):
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

        return video_length, mask_seq, train_state
    
    # print:
    def print_trainable_parameters(self, model):
        trainable_params, all_param = 0, 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        para_info_str = "trainable params: {} || "
        para_info_str += "all params: {} || "
        para_info_str += "trainable%: {}"
        info_lst = [trainable_params, all_param, round(100 * trainable_params / all_param, 4)]
        print(para_info_str.format(*info_lst))


if __name__=="__main__":
    video_fine_tune = VideoFineTune()
    video_fine_tune.train()