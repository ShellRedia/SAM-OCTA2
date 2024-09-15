import os
import cv2
import math
import numpy as np
from PIL import Image

class DisplaySequence:
    def __init__(self):
        self.point_size = 10
        self.colors = np.load("color_list.npy").tolist()

        alpha = 0.5
        self.overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
        self.to_color = lambda x, color: (self.to_3ch(x) * color).astype(dtype=np.uint8)

        self.save_dir = "sample_display"
        os.makedirs(self.save_dir, exist_ok=True)

    def display_a_batch(self, batch, sample_name="display_image"):
        image_seq = batch["images"].transpose((0,2,3,1))
        mask_seq = batch["masks"]
        prompts_dct = batch["prompts"]
        num_of_frame, h, w, c = image_seq.shape
        display_images = []
        for frame_idx in range(num_of_frame):
            image, mask = image_seq[frame_idx], mask_seq[frame_idx]
            colored_mask = np.zeros_like(image)
            for obj_idx in mask:
                colored_mask += self.to_color(mask[obj_idx], self.colors[obj_idx])
            colored_mask = self.overlay(colored_mask, image)
            if frame_idx in prompts_dct:
                for object_id in prompts_dct[frame_idx]:
                    points = prompts_dct[frame_idx][object_id]["points"]
                    labels = prompts_dct[frame_idx][object_id]["labels"]
                    for (x, y), is_positive in zip(points, labels):
                        point_color = (0, 255, 0) if is_positive else (0, 0, 255)
                        cv2.circle(colored_mask, (int(x), int(y)), self.point_size, point_color, -1)
            display_images.append(colored_mask)
        cv2.imwrite("{}/{}.png".format(self.save_dir, sample_name), np.concatenate(display_images, axis=1))
    
    def display_a_batch_prediction(self, image_seq, mask_seq, pred_seq, prompts_dct, save_dir="prediction", sample_name="display_image"):
        os.makedirs(save_dir, exist_ok=True)

        image_seq = image_seq.cpu().numpy().transpose((0,2,3,1))

        num_of_frame, h, w, c = image_seq.shape

        images, gt_images, pred_images, pred_o_images = [], [], [], []

        for frame_idx in range(num_of_frame):
            image, mask, pred = image_seq[frame_idx], mask_seq[frame_idx], pred_seq[frame_idx]
            images.append(image)

            gt_colored_mask, pred_colored_mask = np.zeros_like(image), np.zeros_like(image)
            for obj_idx in mask:
                gt_colored_mask += self.to_color(mask[obj_idx][0].numpy(), self.colors[obj_idx])
                pred_colored_mask += self.to_color(pred[obj_idx], self.colors[obj_idx])
            # gt_colored_mask = self.overlay(gt_colored_mask, image)
            pred_colored_mask_overlayed = self.overlay(pred_colored_mask, image)

            if frame_idx in prompts_dct:
                for object_id in prompts_dct[frame_idx]:
                    points = prompts_dct[frame_idx][object_id]["points"].numpy()[0]
                    labels = prompts_dct[frame_idx][object_id]["labels"].numpy()[0]
                    for (x, y), is_positive in zip(points, labels):
                        point_color = (0, 255, 0) if is_positive else (0, 0, 255)
                        # cv2.circle(gt_colored_mask, (int(x), int(y)), self.point_size, point_color, -1)
                        cv2.circle(pred_colored_mask_overlayed, (int(x), int(y)), self.point_size, point_color, -1)

            gt_images.append(gt_colored_mask)
            pred_images.append(pred_colored_mask)
            pred_o_images.append(pred_colored_mask_overlayed)
            
        gt_concat = np.concatenate(gt_images, axis=1)
        pred_concat = np.concatenate(pred_images, axis=1)
        pred_o_concat = np.concatenate(pred_o_images, axis=1)
        images_concat = np.concatenate(images, axis=1)
        display_image = np.concatenate([gt_concat, pred_concat, pred_o_concat, images_concat], axis=0)
        cv2.imwrite("{}/{}.png".format(save_dir, sample_name), display_image)

    
    def display_a_frame_prediction(self, image, masks, obj_idxs, prompt_points=[]):
        # prompt_points: [[x_0, y_0, pos/neg], [x_1, y_1, pos/neg]]
        colored_mask = np.zeros_like(image)
        for mask, obj_idx in zip(masks, obj_idxs):
            colored_mask += self.to_color(mask, self.colors[obj_idx])
        overlay_image = self.overlay(image, colored_mask)

        for x, y, is_pos in prompt_points:
            point_color = (0, 255, 0) if is_pos else (0, 0, 255)
            cv2.circle(overlay_image, (x, y), self.point_size, point_color, -1)
        
        return overlay_image
    
    def convert_cv2_to_PIL(self, cv_image):
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image_rgb)
    
    def make_gif(self, frames, save_path):
        frames = [self.convert_cv2_to_PIL(x) for x in frames]
        frames[0].save(save_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)                    

