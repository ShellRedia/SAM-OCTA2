import os
import cv2
import random
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import *

overlay = lambda x, y: cv2.addWeighted(x, 0.7, y, 0.3, 0)
to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_color = lambda x, color: (to_3ch(x) * color).astype(dtype=np.uint8)

get_binary_mask = lambda x : np.where((lambda b, g, r: (r != g) | (r != b) | (g != b))(*cv2.split(x)), 255, 0).astype(np.uint8)
get_green_mask  = lambda x: np.where((lambda b,g,r: (g>b+30)&(g>r+30)&(g>50))(*cv2.split(x)), 255, 0).astype(np.uint8)
get_purple_mask = lambda x : np.where((lambda b,g,r: (g == 0)&(r == 255)&(b == 255))(*cv2.split(x)), 255, 0).astype(np.uint8)

normalize_image = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

class Predictor_Simple:
    def __init__(self):
        self.dataset_dir = "datasets/Sparse/OCTA-500/sam2_layers"
        self.annotation_dir = "datasets/Sparse/OCTA-500/sam2_region"
        self.rv_dir = "datasets/Sparse/OCTA-500/RV/Projection"

        self.sample_dir = "datasets/Sparse/OCTA-500/sample"
        
        os.makedirs(self.annotation_dir, exist_ok=True)
        self.layer_image_files = sorted(os.listdir(self.dataset_dir))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        dints_space = TopologyInstance(spatial_dims=2, num_blocks=6, device="cuda")
        model = DiNTS(dints_space=dints_space, in_channels=1, num_classes=1, spatial_dims=2)
        self.model = ModifiedModel(model).to(self.device)

        region_weight_path = "pretrained_weights/dints_region.pth"
        self.model.load_state_dict(torch.load(region_weight_path))
        self.model.eval()
        

    def build_interface(self):
        with gr.Tab("OCTA-500层序标注"):
            brush = gr.Brush(default_size=10, colors = ["rgb(255, 0, 255)", "rgb(0, 255, 0)"], default_color="rgb(0, 255, 0)", color_mode="defaults")
            
            with gr.Row():
                sample_id = gr.Slider(label="样本id", value=10001, minimum=10001, maximum=10500)
                sample_layer_idx= gr.Slider(label="样本切片层索引", value=200, minimum=0, maximum=639)
                sample_layer_name = gr.Text(label="样本切片名", interactive=False)
                annotated_layers_num = gr.Number(label="已标注数量", interactive=False)
                is_saved = gr.Checkbox(label="已保存")

                with gr.Column():
                    prev_annotated_button = gr.Button("前一已标注")
                    next_annotated_button = gr.Button("后一已标注")
                with gr.Column():
                    updated_button = gr.Button("更新")
                    save_button = gr.Button("保存")
                with gr.Column():
                    predict_button = gr.Button("预测标注")
                    clear_button = gr.Button("清除标注")

            with gr.Accordion("批量预测", open=False):
                with gr.Row():
                    layer_range_start = gr.Slider(label="起始层索引", value=200, minimum=0, maximum=639)
                    layer_range_end = gr.Slider(label="结束层索引", value=300, minimum=0, maximum=639)
                    pred_count = gr.Slider(label="预测帧数", value=12, minimum=2, maximum=20)
                    min_interval = gr.Slider(label="最小间隔", value=5, minimum=3, maximum=20)
                    batch_predict_button = gr.Button("批量预测")
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        layer_image = gr.Image(label="1", interactive=False)
                        layer_rv = gr.Image(label="2", interactive=False)
                    with gr.Row():
                        layer_overlayed = gr.Image(label="3", interactive=False)
                        layer_rv_full = gr.Image(label="4", interactive=False)
                with gr.Column():
                    layer_edit = gr.Sketchpad(label="编辑标注", type='numpy', brush=brush, eraser=gr.Eraser(5), interactive=True)
                    annotated_sample_layer_names = gr.Text(label="已标注序列(名称)", interactive=False)
                    annotated_sample_seq = gr.Image(label="已标注序列", interactive=False)
            
            inputs = [sample_id, sample_layer_idx]
            outputs = [sample_layer_idx, sample_layer_name]
            prev_annotated_button.click(self.get_prev_annotated_layer, inputs=inputs, outputs=outputs)
            next_annotated_button.click(self.get_next_annotated_layer, inputs=inputs, outputs=outputs)

            inputs = [sample_id, sample_layer_idx]
            outputs = [sample_layer_name, layer_image, layer_edit, layer_rv, layer_overlayed, layer_rv_full, annotated_sample_layer_names, annotated_sample_seq, annotated_layers_num, is_saved]
            sample_id.change(self.get_layer_images, inputs=inputs, outputs=outputs)
            sample_layer_idx.change(self.get_layer_images, inputs=inputs, outputs=outputs)
            clear_button.click(self.clear_annotation, inputs=inputs, outputs=outputs)

            inputs = [sample_layer_name, layer_image, layer_edit]
            outputs = [layer_rv, layer_overlayed, layer_rv_full, is_saved]
            updated_button.click(self.update_annotation, inputs=inputs, outputs=outputs)
            
            inputs = [sample_id, sample_layer_idx, layer_image, layer_overlayed]
            outputs = [annotated_sample_layer_names, annotated_sample_seq, annotated_layers_num, is_saved]
            save_button.click(self.save_annotation, inputs=inputs, outputs=outputs)

            inputs = [sample_layer_name, layer_image]
            outputs = [layer_rv, layer_overlayed, layer_rv_full, layer_edit, is_saved]
            predict_button.click(self.predict_annotation, inputs, outputs)

            # batch prediction
            inputs = [sample_id, layer_range_start, layer_range_end, pred_count, min_interval]
            outputs = [is_saved]
            batch_predict_button.click(self.batch_predict_annotation, inputs=inputs, outputs=outputs)

    def get_layer_images(self, sample_id, sample_layer_idx):
        sample_volume = np.load("{}/{}.npy".format(self.sample_dir, sample_id))
        layer_image = sample_volume[sample_layer_idx]
        layer_image = to_3ch(normalize_image(layer_image))

        sample_layer_name = "{}_{:0>3}".format(sample_id, sample_layer_idx)

        anno_path = "{}/{}.png".format(self.annotation_dir, sample_layer_name)
        
        layer_anno = np.zeros_like(layer_image[:,:,0])
        if os.path.exists(anno_path):
            layer_anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
            layer_anno = layer_anno[:, layer_anno.shape[0]:]
        layer_overlayed = overlay(layer_image, to_color(layer_anno, (0, 1, 0)))

        rv = cv2.imread("{}/{}.bmp".format(self.rv_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        layer_rv_full = overlay(layer_image, to_color(np.transpose(rv), (1, 1, 0)))

        rv = (np.transpose(rv) / 255 * layer_anno).astype(np.uint8)
        rv_overlayed = overlay(layer_image, to_color(rv, (1, 1, 0)))

        annotated_sample_layer_names, annotated_sample_seq, layer_num = self.get_seq_images(sample_id)

        return sample_layer_name, layer_image, layer_overlayed, rv_overlayed, layer_overlayed, layer_rv_full, annotated_sample_layer_names, annotated_sample_seq, layer_num, False
    
    def update_annotation(self, sample_layer_name, layer_image, layer_edit):
        layer_0 = layer_edit["composite"][:,:,:3].astype(np.uint8)

        binary_mask = get_green_mask(layer_0)
        binary_mask = np.where(binary_mask > 127, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(binary_mask, contours, 255)

        binary_mask -= get_purple_mask(layer_0)

        layer_overlayed = overlay(layer_image, to_color(binary_mask, (0, 1, 0)))

        sample_id = sample_layer_name.split("_")[0]
        rv = cv2.imread("{}/{}.bmp".format(self.rv_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        layer_rv_full = overlay(layer_image, to_color(np.transpose(rv), (1, 1, 0)))

        rv = (np.transpose(rv) / 255 * binary_mask).astype(np.uint8)
        rv_overlayed = overlay(layer_image, to_color(rv, (1, 1, 0)))

        return rv_overlayed, layer_overlayed, layer_rv_full, False
    
    def save_annotation(self, sample_id, sample_layer_idx, layer_image, layer_overlayed):
        sample_layer_name = "{}_{:0>3}".format(sample_id, sample_layer_idx)
        anno_path = "{}/{}.png".format(self.annotation_dir, sample_layer_name)
        binary_mask = get_green_mask(layer_overlayed)

        filled = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5,5)))

        cv2.imwrite(anno_path, np.concatenate([layer_image[:,:,0], filled], axis=1))

        return *self.get_seq_images(sample_id), True
    
    @torch.no_grad()
    def predict_annotation(self, sample_layer_name, layer_image):
        input_image = layer_image[:,:,:1].transpose((2,0,1)) / 255
        input_image = np.expand_dims(input_image, axis=0)

        input_image = torch.from_numpy(input_image).to(self.device).float()
        
        binary_mask = self.model(input_image).detach().cpu().numpy()[0][0]
        binary_mask = np.where(binary_mask>0.7, 255, 0)

        layer_edit = layer_overlayed = overlay(layer_image, to_color(binary_mask, (0, 1, 0)))

        sample_id = sample_layer_name.split("_")[0]
        rv = cv2.imread("{}/{}.bmp".format(self.rv_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        layer_rv_full = overlay(layer_image, to_color(np.transpose(rv), (1, 1, 0)))

        rv = (np.transpose(rv) / 255 * binary_mask).astype(np.uint8)
        rv_overlayed = overlay(layer_image, to_color(rv, (1, 1, 0)))
        
        return rv_overlayed, layer_overlayed, layer_rv_full, layer_edit, False
    
    def clear_annotation(self, sample_id, sample_layer_idx):
        sample_layer_name = "{}_{:0>3}".format(sample_id, sample_layer_idx)
        anno_path = "{}/{}.png".format(self.annotation_dir, sample_layer_name)
        if os.path.exists(anno_path): os.remove(anno_path)
        return self.get_layer_images(sample_id, sample_layer_idx)
    
    def get_seq_images(self, sample_id):
        annotated_sample_seq = []
        annotated_sample_layer_names = []

        for sample_layer_file in sorted(os.listdir(self.annotation_dir)):
            if sample_layer_file.split("_")[0] == str(sample_id):
                layer_anno_img = cv2.imread("{}/{}".format(self.annotation_dir, sample_layer_file), cv2.IMREAD_GRAYSCALE)
                h = layer_anno_img.shape[0]
                overlayed = overlay(to_3ch(layer_anno_img[:,:h]), to_color(layer_anno_img[:,h:], (0,1,0)))
                annotated_sample_seq.append(overlayed)
                annotated_sample_layer_names.append(sample_layer_file.split("_")[1][:-4])
        annotated_sample_layer_names = ",".join(annotated_sample_layer_names)
        return annotated_sample_layer_names, np.concatenate(annotated_sample_seq, axis=1) if annotated_sample_seq else None, len(annotated_sample_seq)
    
    def get_prev_annotated_layer(self, sample_id, sample_layer_idx):
        for sample_layer_file in sorted(os.listdir(self.annotation_dir), reverse=True):
            sid, lid = sample_layer_file.split("_")
            lid = int(lid[:-4])
            if sid == str(sample_id) and int(lid) < sample_layer_idx:
                return int(lid), "{}_{}".format(sample_id, lid)
        return sample_layer_idx, "{}_{}".format(sample_id, sample_layer_idx)

    def get_next_annotated_layer(self, sample_id, sample_layer_idx):
        for sample_layer_file in sorted(os.listdir(self.annotation_dir)):
            sid, lid = sample_layer_file.split("_")
            lid = int(lid[:-4])
            if sid == str(sample_id) and int(lid) > sample_layer_idx:
                return int(lid), "{}_{}".format(sample_id, lid)
        return sample_layer_idx, "{}_{}".format(sample_id, sample_layer_idx)
    
    @torch.no_grad()
    def batch_predict_annotation(self, sample_id, layer_range_start, layer_range_end, pred_count, min_interval):
        S, E, N, D = layer_range_start, layer_range_end, pred_count, min_interval

        excess = (E - S) - (N - 1) * D

        weights = [random.uniform(0.8, 1.2) for _ in range(N - 1)]
        scale = excess / sum(weights)

        idx_lst = [S]
        curr = S
        for i in range(N - 1):
            step = D + weights[i] * scale
            curr += step
            idx_lst.append(round(curr))

        idx_lst[-1] = E

        for sample_layer_idx in idx_lst:
            layer_image = self.get_layer_images(sample_id, sample_layer_idx)[1]

            input_image = layer_image[:,:,:1].transpose((2,0,1)) / 255
            input_image = np.expand_dims(input_image, axis=0)

            input_image = torch.from_numpy(input_image).to(self.device).float()
            
            binary_mask = self.model(input_image).detach().cpu().numpy()[0][0]
            binary_mask = np.where(binary_mask>0.7, 255, 0)

            sample_layer_name = "{}_{:0>3}".format(sample_id, sample_layer_idx)
            anno_path = "{}/{}.png".format(self.annotation_dir, sample_layer_name)
            cv2.imwrite(anno_path, np.concatenate([layer_image[:,:,0], binary_mask], axis=1))
        
        return True

if __name__ == "__main__":
    predictor = Predictor_Simple()
    with gr.Blocks() as launcer:
        predictor.build_interface()
    launcer.launch()