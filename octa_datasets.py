from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from scipy.ndimage import label, center_of_mass

from scipy.ndimage import label, find_objects
from skimage.measure import regionprops

from collections import Counter

import random
from random import randint, choice, shuffle

from tqdm import tqdm

from prompts import PromptGeneration
import imgaug.augmenters as iaa

from display import DisplaySequence


class OCTA_Dataset_Layer_Sparse_Annotation_Training(Dataset):
    def __init__(self, image_sz=512):
        self.data_dir = "datasets/OCTA500/RV_Annotation"
        
        self.samples, self.masks = [], []
        for file_name in sorted(os.listdir(self.data_dir)):
            sample = cv2.imread("{}/{}".format(self.data_dir, file_name), cv2.IMREAD_GRAYSCALE)
            sample = cv2.resize(sample, (image_sz * 2, image_sz))
            self.samples.append(sample[:, :image_sz])
            self.masks.append(sample[:, image_sz:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        process = lambda x : np.array([x], dtype=np.float32) / 255
        sample, mask = self.samples[index], self.masks[index]
        
        return process(sample), process(mask)

class OCTA_Dataset_Layer_Sparse_Annotation_Prediction(Dataset):
    def __init__(self, image_sz=512):
        data_dir = "datasets/OCTA500/RV_SampleLayer"
        sample_names = [int(x) for x in os.listdir(data_dir)]
        self.image_sz = image_sz
        self.sample_identifiers, self.sample_path = [], []

        for sample_name in sample_names:
            sample_dir = "{}/{}".format(data_dir, sample_name)
            for layer_name in os.listdir(sample_dir):
                self.sample_identifiers.append((sample_name, layer_name))
                self.sample_path.append("{}/{}".format(sample_dir, layer_name))
    
    def __len__(self):
        return len(self.sample_identifiers)
    
    def __getitem__(self, index):
        process = lambda x : np.array([x], dtype=np.float32) / 255
        image = cv2.imread(self.sample_path[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_sz, self.image_sz))

        return process(image), self.sample_identifiers[index]
    
class ShuffleChannels(iaa.meta.Augmenter):
    def __init__(self, name=None, random_state=None):
        super().__init__(name=name, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return [image[..., random_state.permutation(image.shape[-1])] for image in images]

    def get_parameters(self):
        return []
    
class RandomChannelToThree(iaa.meta.Augmenter):
    def __init__(self, name=None, random_state=None):
        super().__init__(name=name, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        augmented_images = []
        for image in images:
            channel = random_state.randint(0, image.shape[-1])
            augmented_image = np.stack([image[..., channel]] * 3, axis=-1)
            augmented_images.append(augmented_image)
        return augmented_images

    def get_parameters(self):
        return []
    
seq = iaa.Sequential([
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
    iaa.Sometimes(0.1, ShuffleChannels()),
    iaa.Sometimes(0.1, RandomChannelToThree()),
], random_order=True) # apply augmenters in random order

# seq = iaa.Sequential([
#     iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1))),
#     iaa.Sometimes(0.1, iaa.LinearContrast((0.75, 1.5))),
#     # iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
#     iaa.Sometimes(0.1, iaa.Rotate(rotate=(-10, 10), mode='constant')),
#     iaa.Sometimes(0.1, iaa.Sharpen((0.0, 0.5))),
#     # iaa.Sometimes(0.1, iaa.ElasticTransformation(sigma=15)),
#     iaa.Sometimes(0.1, iaa.ImpulseNoise(p=0.05)),
# ], random_order=True) # apply augmenters in random order

def data_augmentation(image, mask):
    to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
    is_image_2ch, is_mask_2ch = bool(len(image.shape) == 2), bool(len(mask.shape) == 2)
    if is_image_2ch: image = to_3ch(image)
    if is_mask_2ch: mask = to_3ch(mask)
    images, masks = np.expand_dims(image, axis=0), np.expand_dims(mask, axis=0)
    images_aug, masks_aug = seq(images=images, segmentation_maps=masks)
    images_aug, masks_aug = images_aug[0], masks_aug[0]
    if is_image_2ch: images_aug = images_aug[:,:,0]
    if is_mask_2ch: masks_aug = masks_aug[:,:,0]
    return images_aug, masks_aug

class OCTA_Dataset_SAM2_Sequence(Dataset):
    def __init__(self, 
                 dataset_name="3M", 
                 label_type="RV", 
                 subset="train", 
                 frame_length=5, 
                 prompt_frames=2, 
                 prompt_num=2,
                 is_local=True,
                 prompt_type="point"
                ):
        self.dataset_name = "OCTA-500" if dataset_name == "3M" or dataset_name == "6M" else dataset_name
        self.label_type = label_type
        self.subset = subset
        self.frame_length = frame_length
        self.prompt_frames = prompt_frames
        self.prompt_num = prompt_num
        self.is_local = is_local
        self.prompt_type = prompt_type
        
        if dataset_name == "3M":
            self.sample_ids = list({
                "train":range(10301, 10441),
                "val":range(10441, 10451),
                "test":range(10451, 10501)}[subset])
        elif dataset_name == "6M": 
            self.sample_ids = list({
                "train":range(10001, 10181),
                "val":range(10181, 10201), 
                "test":range(10201, 10301)}[subset])
        elif dataset_name == "Soul":
            self.sample_ids = list({
                "train":range(101, 128), 
                "val":range(128, 135), 
                "test":range(128, 135)}[subset])

        self.sample_objs = [] # list for -> (sample_id, obj_id)
        
        self.sample_seq_dct = {}
        self.sample_obj_color = {}
        self.sample_obj_neg_colors = {}

        for sample_id in self.sample_ids: self.get_objects_by_sample_id(sample_id)

        random_seed = 0 if subset=="train" else 42
        self.pg = PromptGeneration(random_seed=random_seed)
        self.cached_batch = {}

        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

    def get_objects_by_sample_id(self, sample_id):
        region_dir = "Local" if self.is_local else "Global"
        sample_path = "/".join([self.dataset_name, region_dir, self.label_type, str(sample_id)])
        sample_path = "datasets/Sequence/{}.png".format(sample_path)

        if os.path.exists(sample_path):
            sample_seq_image = cv2.imread(sample_path)
            self.sample_seq_dct[sample_id] = sample_seq_image

            h = sample_seq_image.shape[0]
            pixels = sample_seq_image[h//2:].reshape(-1, 3)
            non_black = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
            unique_colors = np.unique(non_black, axis=0)

            self.sample_obj_neg_colors[sample_id] = unique_colors

            for object_idx, color in enumerate(unique_colors):
                self.sample_objs.append((sample_id, object_idx))
                self.sample_obj_color[(sample_id, object_idx)] = color
    
    def get_empty_prompt(self, object_id=0):
        xd, yd = {"RV":(1, -1), "FAZ":(-1, 1), "Artery":(-1,-1), "Vein":(1, 1)}[self.label_type]
        fixed_coord = [xd * 50 + 100, yd * 50 + 100]
        return {0:{object_id:{"points":np.array([fixed_coord], dtype=np.float32),
                              "labels":np.array([1], dtype=np.int32)}}}
    
    def octa_data_augmentation(self, sample_seq):
        augmented_sample_seq = []
        for image, mask in sample_seq:
            augmented_sample_seq.append(data_augmentation(image, mask))
        return augmented_sample_seq

    def load_sequence(self, sample_id, object_id):
        sample_seq_image = self.sample_seq_dct[sample_id]
        obj_color = self.sample_obj_color[(sample_id, object_id)]
        neg_colors = self.sample_obj_neg_colors[sample_id]

        h, w = sample_seq_image.shape[:2]
        sz = h // 2
        seq_len = w // sz

        target_size = (1024, 1024)

        layer_sequence = [sample_seq_image[:, i*sz:(i+1)*sz] for i in range(seq_len)]

        if self.subset == "train":
            if self.dataset_name == "Soul":
                frame_length = randint(3, seq_len)
            elif self.dataset_name == "OCTA-500":
                frame_length = randint(4, seq_len)

            num_of_prompt_frames = randint(1, min(3, frame_length))

            start = randint(0, len(layer_sequence) - frame_length)
            selected_layer_sequence = layer_sequence[start:start + frame_length]

            selected_layer_sequence = [(x[:sz], x[sz:]) for x in selected_layer_sequence]
            selected_layer_sequence = self.octa_data_augmentation(selected_layer_sequence)
        else:
            num_of_prompt_frames = self.prompt_frames
            if self.label_type == "FAZ":
                start = (len(layer_sequence) - self.frame_length) // 2
                selected_layer_sequence = layer_sequence[start : start + self.frame_length]
            else:
                selected_layer_sequence = layer_sequence[:self.frame_length] # self.frame_length, self.prompt_frames

            selected_layer_sequence = [(x[:sz], x[sz:]) for x in selected_layer_sequence]

        layer_image_seq = [x[0] for x in selected_layer_sequence]
        layer_mask_seq = [x[1] for x in selected_layer_sequence]

        # get_prompt_frame_idxs
        prompt_frame_idxs = []
        for i, mask in enumerate(layer_mask_seq):
            if np.any(np.all(mask==obj_color, axis=-1)):
                prompt_frame_idxs.append(i)
        
        prompt_idxs = {
            0:[],
            1:[0], 
            2:[0, len(prompt_frame_idxs)-1], 
            3:[0, len(prompt_frame_idxs) // 2, len(prompt_frame_idxs)-1]
        }[min(len(prompt_frame_idxs), num_of_prompt_frames)]

        prompt_idxs = [prompt_frame_idxs[i] for i in prompt_idxs]

        if self.is_local:
            layer_mask_seq_pos = [np.all(x == obj_color, axis=-1).astype(np.uint8) * 255 for x in layer_mask_seq]
        else:
            layer_mask_seq_pos = [np.all(x != np.zeros(3, int), axis=-1).astype(np.uint8) * 255 for x in layer_mask_seq]
            prompt_idxs = prompt_idxs[:1]

        layer_mask_seq_neg = []

        for x in layer_mask_seq:
            neg_canvas = np.zeros_like(layer_mask_seq_pos[0], dtype=np.uint8)
            for color in neg_colors:
                if not np.array_equal(color, obj_color):
                    neg_canvas += np.all(x == color, axis=-1)
            layer_mask_seq_neg.append(np.where(neg_canvas > 0, 255, 0).astype(np.uint8))
        

        image_lst, layer_mask_lst, neg_layer_mask_lst = np.array(layer_image_seq), np.array(layer_mask_seq_pos), np.array(layer_mask_seq_neg)

        image_seq, mask_seq, prompts_dct = [], [], {}
            
        for frame_idx in range(len(image_lst)):  
            layer_image, layer_mask, layer_neg_mask = image_lst[frame_idx], layer_mask_lst[frame_idx], neg_layer_mask_lst[frame_idx]

            layer_image = cv2.resize(layer_image, target_size)
            layer_mask = cv2.resize(layer_mask, target_size)
            layer_neg_mask = cv2.resize(layer_neg_mask, target_size)

            layer_mask = np.where(layer_mask > 0, 255, 0)
            layer_neg_mask = np.where(layer_neg_mask > 0, 255, 0)

            image_seq.append(layer_image)

            if frame_idx in prompt_idxs and np.sum(layer_mask) > 0:
                prompts_dct[frame_idx] = {}
                prompts_dct[frame_idx][object_id] = {}
 
                prompt_num = randint(1, 4) if self.subset == "train" else self.prompt_num

                coord_pos = self.pg.get_prompt_points(layer_mask, prompt_num)
                # coord_neg = self.pg.get_prompt_points(layer_neg_mask, 1)

                if self.is_local:
                    prompts_dct[frame_idx][object_id]["points"] = np.array(coord_pos, dtype=np.float32)
                    prompts_dct[frame_idx][object_id]["labels"] = np.array([1] * len(coord_pos), dtype=np.int32)
                    # prompts_dct[frame_idx][object_id]["points"] = np.array(coord_pos + coord_neg, dtype=np.float32)
                    # prompts_dct[frame_idx][object_id]["labels"] = np.array([1] * len(coord_pos) + [0] * len(coord_neg), dtype=np.int32)
                else:
                    xd, yd = {"RV":(1, -1), "FAZ":(-1, 1), "Artery":(-1,-1), "Vein":(1, 1)}[self.label_type]
                    fixed_coord = [xd * 50 + 100, yd * 50 + 100]
                    prompts_dct[frame_idx][object_id]["points"] = np.array([fixed_coord], dtype=np.float32)
                    prompts_dct[frame_idx][object_id]["labels"] = np.array([1], dtype=np.int32)

            mask_seq.append({object_id:layer_mask / 255})

        image_seq = np.array(image_seq)

        if not prompts_dct: 
            prompts_dct = self.get_empty_prompt(object_id)


        return image_seq, mask_seq, prompts_dct
 
    def __len__(self):
        return len(self.sample_objs)
    
    def __getitem__(self, index):
        # prompts_lst -> coords(x, y), pos/neg, object_id: [x, y, 1/0, 0...n], four elements tuple
        sample_id, object_id = self.sample_objs[index]
        image_seq, mask_dct_seq, prompts_dct = self.load_sequence(sample_id, object_id)
        
        # format batch
        sample_name = "{}_{:0>2}".format(sample_id, object_id)

        if sample_name in self.cached_batch:
            batch = self.cached_batch[sample_name]
        else:
            image_seq = np.array(image_seq).transpose((0,3,1,2))
            mask_seq = mask_dct_seq

            batch = {
                "sample_name": sample_name,
                "images": image_seq,
                "masks": mask_seq,
                "prompts": prompts_dct
            }

            if self.subset != "train": self.cached_batch[sample_name] = batch

        return batch


class OCTA_Dataset_SAM2_Single(Dataset):
    def __init__(self,
                 dataset_name="3M", 
                 label_type="RV",
                 subset="train",
                 prompt_num=2,
                 is_local=True
                ):
        self.dataset_name = "OCTA-500" if dataset_name == "3M" or dataset_name == "6M" else dataset_name
        self.label_type = label_type
        self.subset = subset
        self.prompt_num = prompt_num
        self.is_local = is_local
        
        if dataset_name == "3M":
            self.sample_ids = list({
                "train":range(10301, 10441),
                "val":range(10441, 10451), 
                "test":range(10451, 10501)}[subset])
        elif dataset_name == "6M": 
            self.sample_ids = list({
                "train":range(10001, 10181),
                "val":range(10181, 10201), 
                "test":range(10201, 10301)}[subset])
        elif dataset_name == "ROSE":
            self.sample_ids = list({
                "train":range(101, 130), 
                "val":range(131, 140), 
                "test":range(131, 140)}[subset])

        self.cached_batch = {}
        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

    def load_sample(self, sample_id):
        sample_file = "/".join([self.dataset_name, self.label_type, str(sample_id)])
        sample_file = "datasets/Single/{}.png".format(sample_file)
        sample_image = cv2.imread(sample_file, cv2.IMREAD_COLOR)
        h, w = sample_image.shape[:2]
        image, mask = sample_image[:, :h], sample_image[:, h:]

        target_size = (1024, 1024)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        if self.subset == "train": image, mask = data_augmentation(image, mask)

        image_seq = np.expand_dims(image.transpose((2,0,1)), axis=0)
        mask = np.where(mask > 0, 1, 0)
        mask_seq = [{0: mask[:,:,0]}]
        
        if self.is_local:
            pass
        else:
            prompts_dct = {0:{0:{"points":np.array([[100, 100]], dtype=np.float32),
                                "labels":np.array([1], dtype=np.int32)}}}
        
        return image_seq, mask_seq, prompts_dct
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        # prompts_lst -> coords(x, y), pos/neg, object_id: [x, y, 1/0, 0...n], four elements tuple
        sample_id = self.sample_ids[index]

        image_seq, mask_dct_seq, prompts_dct = self.load_sample(sample_id)

        if sample_id in self.cached_batch:
            batch = self.cached_batch[sample_id]
        else:
            mask_seq = mask_dct_seq

            batch = {
                "sample_name": sample_id,
                "images": image_seq,
                "masks": mask_seq,
                "prompts": prompts_dct
            }

            if self.subset != "train": self.cached_batch[sample_id] = batch

        return batch

class OCTA_Dataset_monai(Dataset):
    def __init__(self,
                dataset_name="3M", 
                label_type="RV", 
                data_type="Sequence",
                subset="train", 
                frame_length=8):
        self.dataset_name = "OCTA-500" if dataset_name == "3M" or dataset_name == "6M" else dataset_name
        self.label_type = label_type
        self.data_type = data_type
        self.subset = subset
        self.frame_length = frame_length
        
        if dataset_name == "3M":
            self.sample_ids = list({"train":list(range(10301, 10441)),
                                     "val":range(10441, 10451), "test":range(10451, 10501)}[subset])
        elif dataset_name == "6M": 
            self.sample_ids = list({"train":list(range(10001, 10181)),
                                     "val":range(10181, 10201), "test":range(10201, 10301)}[subset])
        elif dataset_name == "ROSE":
            self.sample_ids = list({"train":range(101, 130), "val":range(131, 140), "test":range(131, 140)}[subset])

        elif dataset_name == "Soul":
            self.sample_ids = list({"train":range(101, 128), "val":range(128, 129), "test":range(128, 135)}[subset])

        self.load_sample = {
            "single":self.load_sample_single,
            "sequence":self.load_sample_sequence
        }[data_type]
    
    def load_sample_single(self, sample_id):
        sample_file = "datasets/Single/{}/{}/{}.png".format(self.dataset_name, self.label_type, sample_id)

        sample_image = cv2.imread(sample_file, cv2.IMREAD_COLOR)
        h, w = sample_image.shape[:2]
        image, mask = sample_image[:, :h], sample_image[:, h:]

        target_size = (1024, 1024)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        if self.subset == "train": image, mask = data_augmentation(image, mask)

        image = image.transpose((2,0,1)) / 255
        mask = np.where(mask > 0, 1, 0)[:,:,:1].transpose((2,0,1))

        return image, mask

    def load_sample_sequence(self, sample_id):
        sample_file = "datasets/{}/{}_global/{}.png".format(self.dataset_name, self.label_type, sample_id)

        sample_image = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
        
        h, w = sample_image.shape[:2]
        sz = h // 2
        
        sample_seq = sample_image[:, :sz*self.frame_length]

        frame_len = min(self.frame_length, w // sz)

        sample_seq = [(sample_seq[:sz, i*sz:(i+1)*sz], sample_seq[sz:, i*sz:(i+1)*sz]) for i in range(frame_len)]

        resize = lambda x: cv2.resize(x, (1024, 1024))
 
        sample_seq = [(resize(x),resize(y)) for x, y in sample_seq]

        while len(sample_seq) < self.frame_length:
            sample_seq.append((np.zeros((1024, 1024), dtype=np.uint8), np.zeros((1024, 1024), dtype=np.uint8)))

        if self.subset == "train": 
            sample_seq = [data_augmentation(x, y) for x, y in sample_seq]

        image = np.dstack([x[0] for x in sample_seq]).transpose((2,0,1)) / 255
        mask = np.dstack([x[1] for x in sample_seq]).transpose((2,0,1))
        mask = np.where(mask > 0, 1, 0)

        

        return image, mask

    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        image, mask = self.load_sample(sample_id)
        return image, mask, sample_id


# if __name__=="__main__":
#     pass
#     octa_dataset = OCTA_Dataset_SAM2_Evaluation(label_type="FAZ")
#     # for sample_name in tqdm(octa_dataset.sample_names):
#     #     octa_dataset.mark_rv_objects(sample_name)

#     ds = DisplaySequence()

#     for idx in tqdm(range(len(octa_dataset))):
#         batch = octa_dataset[idx]
#         # print(batch["images"].shape, batch["images"].max(), batch["images"].dtype)
#         # print(batch["prompts"])
#         ds.display_a_batch(batch, "display_{}".format(idx))
#         break
        

