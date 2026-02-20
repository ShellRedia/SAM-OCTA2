from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import albumentations as alb

from scipy.ndimage import label, center_of_mass

from scipy.ndimage import label, find_objects
from skimage.measure import regionprops

from collections import Counter

from random import randint, sample, choice
from tqdm import tqdm

from prompts import PromptGeneration
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
    

class OCTA_Dataset_SAM2_Training(Dataset):
    def __init__(self, validation_mode=False, fov="3M", label_type="RV"):
        self.validation_mode = validation_mode
        self.label_type = label_type

        self.valid_batch_dct = {}
        
        self.load_sequence = self.load_sequence_rv
        if label_type == "FAZ": self.load_sequence = self.load_sequence_faz
        

        self.layer_root_dir = "datasets/OCTA500/{}_LayerSequence".format(label_type)
        self.colored_rv_dir = "datasets/OCTA500/Colored_RV"
        
        self.sample_names = sorted(os.listdir(self.layer_root_dir))

        subset = set(range(10001, 10181)) if fov == "6M" else set(range(10301, 10441))

        self.pg = PromptGeneration(random_seed=0, neg_range=(3, 9))

        if validation_mode:
            subset = set(range(10181, 10201)) if fov == "6M" else set(range(10441, 10451))
            self.pg = PromptGeneration(random_seed=42, neg_range=(3, 9))
        
        self.sample_names = [x for x in self.sample_names if int(x) in subset]

        self.select_intervaled_sequence = lambda layer_sequence, n : np.array(layer_sequence)[np.linspace(0, len(layer_sequence)-1, n, dtype=int)]
        self.get_layer_sequence = lambda sample_name : sorted([x for x in os.listdir("{}/{}".format(self.layer_root_dir, sample_name)) if ".png" in x])

        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

        probability = 0.5
        self.transform = alb.Compose([
            alb.SafeRotate(limit=10, p=probability),
            alb.HorizontalFlip(p=probability),
        ])
    
    
    def random_crop_sequence(self, layer_sequence, n):
        beg = randint(0, len(layer_sequence) - n + 1)
        return layer_sequence[beg:beg + n]
    
    def load_sequence_rv(self, sample_name, selected_layer_sequence, prompt_idxs):
        sample_dir = "{}/{}".format(self.layer_root_dir, sample_name)

        target_size = (1024, 1024)

        image_seq, mask_seq, layer_mask_seq = [], [], []

        proposal_objects = set()

        for idx in prompt_idxs:
            layer_name = selected_layer_sequence[idx][:-4]
            sample_objects = np.load("{}/{}/objects.npy".format(sample_dir, layer_name))
            proposal_objects |= set(sample_objects.tolist())
        
        object_id = choice(list(proposal_objects))
        
        sequence_len = len(selected_layer_sequence)

        for frame_idx, layer_file in enumerate(selected_layer_sequence):
            layer_name = layer_file[:-4]
            layer_image = cv2.imread("{}/{}".format(sample_dir, layer_file), cv2.IMREAD_GRAYSCALE)
            _, w = layer_image.shape
            layer_image = cv2.resize(layer_image[:,3*w//4:], target_size)
            layer_mask = np.zeros(target_size)

            layer_mask_path = "{}/{}/{:0>2}.png".format(sample_dir, layer_name, object_id)
            if os.path.exists(layer_mask_path):
                layer_mask = cv2.imread(layer_mask_path, cv2.IMREAD_GRAYSCALE)
                layer_mask = cv2.resize(layer_mask, target_size)

            layer_mask_seq.append(layer_mask)
            image_seq.append(layer_image)

        image_seq = np.array(image_seq).transpose((1,2,0))
        layer_mask_seq = np.array(layer_mask_seq).transpose((1,2,0))

        if not self.validation_mode:
            transformed = self.transform(**{"image": image_seq, "mask": layer_mask_seq})
            image_seq, layer_mask_seq = transformed["image"], transformed["mask"]

        image_lst, layer_mask_lst = image_seq.transpose((2,0,1)), layer_mask_seq.transpose((2,0,1))

        image_seq = []

        prompts_dct = {}
            
        for frame_idx in range(sequence_len):  
            layer_mask = layer_mask_lst[frame_idx]
            layer_image = image_lst[frame_idx]
            image_seq.append(self.to_3ch(layer_image))

            if frame_idx in prompt_idxs and np.sum(layer_mask) > 0:
                prompts_dct[frame_idx] = {}
                prompts_dct[frame_idx][object_id] = {}
                ppp, ppn = randint(1, 10), randint(0, 6)

                negative_region = self.pg.search_negative_region_numpy(layer_mask)

                coord_positive = [[y, x] for x, y in np.argwhere(layer_mask > 0)]
                coord_negative = [[y, x] for x, y in np.argwhere(negative_region > 0)]

                prompts_dct[frame_idx][object_id]["points"] = self.pg.sample_points_from_regions(coord_positive, ppp)
                prompts_dct[frame_idx][object_id]["points"] += self.pg.sample_points_from_regions(coord_negative, ppn)

                prompts_dct[frame_idx][object_id]["labels"] = [1] * ppp + [0] * ppn

                # convert to correct numpy type
                prompts_dct[frame_idx][object_id]["points"] = np.array(prompts_dct[frame_idx][object_id]["points"], dtype=np.float32)
                prompts_dct[frame_idx][object_id]["labels"] = np.array(prompts_dct[frame_idx][object_id]["labels"], dtype=np.int32)
            
            mask_seq.append({object_id:layer_mask / 255})

        image_seq = np.array(image_seq)

        return image_seq, mask_seq, prompts_dct
    
    def load_sequence_faz(self, sample_name, selected_layer_sequence, prompt_idxs):
        sample_dir = "{}/{}".format(self.layer_root_dir, sample_name)

        target_size = (1024, 1024)

        image_seq, mask_seq, layer_mask_seq = [], [], []

        sequence_len = len(selected_layer_sequence)

        for frame_idx, layer_file in enumerate(selected_layer_sequence):
            layer_image_concat = cv2.imread("{}/{}".format(sample_dir, layer_file), cv2.IMREAD_GRAYSCALE)
            _, w = layer_image_concat.shape
     
            layer_image = cv2.resize(layer_image_concat[:, w//3:2*w//3], target_size)
            layer_mask = cv2.resize(layer_image_concat[:, 2*w//3:], target_size)

            layer_mask_seq.append(layer_mask)
            image_seq.append(layer_image)

        image_seq = np.array(image_seq).transpose((1,2,0))
        layer_mask_seq = np.array(layer_mask_seq).transpose((1,2,0))

        if not self.validation_mode:
            transformed = self.transform(**{"image": image_seq, "mask": layer_mask_seq})
            image_seq, layer_mask_seq = transformed["image"], transformed["mask"]

        image_lst, layer_mask_lst = image_seq.transpose((2,0,1)), layer_mask_seq.transpose((2,0,1))

        image_seq = []

        prompts_dct = {}

        object_id = 0
            
        for frame_idx in range(sequence_len):  
            layer_mask = layer_mask_lst[frame_idx]
            layer_image = image_lst[frame_idx]
            image_seq.append(self.to_3ch(layer_image))

            if frame_idx in prompt_idxs and np.sum(layer_mask) > 0:
                prompts_dct[frame_idx] = {}
                prompts_dct[frame_idx][object_id] = {}
                ppp, ppn = randint(1, 10), randint(0, 6)

                coord_positive, coord_negative = self.pg.get_prompt_points(layer_mask, ppp, ppn)

                prompts_dct[frame_idx][object_id]["points"] = coord_positive
                prompts_dct[frame_idx][object_id]["points"] += coord_negative 

                prompts_dct[frame_idx][object_id]["labels"] = [1] * ppp + [0] * ppn

                # convert to correct numpy type
                prompts_dct[frame_idx][object_id]["points"] = np.array(prompts_dct[frame_idx][object_id]["points"], dtype=np.float32)
                prompts_dct[frame_idx][object_id]["labels"] = np.array(prompts_dct[frame_idx][object_id]["labels"], dtype=np.int32)
            
            mask_seq.append({object_id:layer_mask / 255})

        image_seq = np.array(image_seq)

        return image_seq, mask_seq, prompts_dct
 
    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, index):
        # prompts_lst -> coords(x, y), pos/neg, object_id: [x, y, 1/0, 0...n], four elements tuple
        sample_name = self.sample_names[index]

        if sample_name in self.valid_batch_dct and self.validation_mode:
            return self.valid_batch_dct[sample_name]
        
        layer_sequence = self.get_layer_sequence(sample_name)

        training_frame_length, num_of_prompt_frames = randint(4, 8), randint(1, 3)

        cropped_layer_sequence = self.random_crop_sequence(layer_sequence, randint(training_frame_length, len(layer_sequence)))
        selected_layer_sequence = self.select_intervaled_sequence(cropped_layer_sequence, training_frame_length)

        prompt_idxs = {1:[0], 2:[0, training_frame_length-1], 3:[0, training_frame_length // 2, training_frame_length-1]}[num_of_prompt_frames]
        image_seq, mask_dct_seq, prompts_dct = self.load_sequence(sample_name, selected_layer_sequence, prompt_idxs)

        # Crop out several frames of video ...
        image_seq = np.array(image_seq).transpose((0,3,1,2))
        mask_seq = mask_dct_seq

        batch = {
            "sample_name": sample_name,
            "images": image_seq,
            "masks": mask_seq,
            "prompts": prompts_dct
        }

        if self.validation_mode: 
            self.valid_batch_dct[sample_name] = batch

        return batch

class OCTA_Dataset_SAM2_Evaluation(Dataset):
    def __init__(self, fov="3M", label_type="RV", frame_length=4, num_of_prompt_frames=1, positive_points=1, negative_points=0):
        self.label_type = label_type
        self.frame_length = frame_length
        self.num_of_prompt_frames = num_of_prompt_frames
        self.ppp = positive_points
        self.ppn = negative_points

        self.load_sequence = self.load_sequence_rv
        if label_type == "FAZ": self.load_sequence = self.load_sequence_faz

        self.layer_root_dir = "datasets/OCTA500/{}_LayerSequence".format(label_type)
        self.colored_rv_dir = "datasets/OCTA500/Colored_RV"

        subset = set(range(10451, 10501)) if fov == "3M" else set(range(10201, 10301))

        colors = [tuple(x) for x in np.load("color_list.npy")]
        self.colors_dct = dict(zip(colors, range(len(colors))))

        self.pg = PromptGeneration(random_seed=42, neg_range=(3, 9))

        self.select_intervaled_sequence = lambda layer_sequence, n : np.array(layer_sequence)[np.linspace(0, len(layer_sequence)-1, n, dtype=int)]
        self.get_layer_sequence = lambda sample_name : sorted([x for x in os.listdir("{}/{}".format(self.layer_root_dir, sample_name)) if ".png" in x])

        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

        self.batch_dct_lst = []

        sample_names = sorted([x for x in os.listdir(self.layer_root_dir) if int(x) in subset])

        for sample_name in tqdm(sample_names):
            self.load_batches_of_sample(sample_name)
    
    def load_batches_of_sample(self, sample_name):
        layer_sequence = self.get_layer_sequence(sample_name)

        selected_layer_sequence = self.select_intervaled_sequence(layer_sequence, self.frame_length)
        prompt_idxs = {1:[0], 2:[0, self.frame_length-1], 3:[0, self.frame_length // 2, self.frame_length-1]}[self.num_of_prompt_frames]

        sample_dir = "{}/{}".format(self.layer_root_dir, sample_name)
        
        proposal_objects = [0]
        if self.label_type == "RV":
            proposal_objects = set()
            for idx in prompt_idxs:
                layer_name = selected_layer_sequence[idx][:-4]
                sample_objects = np.load("{}/{}/objects.npy".format(sample_dir, layer_name))
                proposal_objects |= set(sample_objects.tolist())
        
        for object_id in proposal_objects:
            image_seq, mask_dct_seq, prompts_dct = self.load_sequence(sample_name, selected_layer_sequence, prompt_idxs, object_id)

            batch_dct = {
                "sample_name": sample_name,
                "images": np.array(image_seq).transpose((0,3,1,2)),
                "masks": mask_dct_seq,
                "prompts": prompts_dct
            }

            self.batch_dct_lst.append(batch_dct)
    
    def load_sequence_rv(self, sample_name, selected_layer_sequence, prompt_idxs, object_id):
        sample_dir = "{}/{}".format(self.layer_root_dir, sample_name)

        target_size = (1024, 1024)

        image_seq, mask_seq = [], []

        prompts_dct = {}

        for frame_idx, layer_file in enumerate(selected_layer_sequence):
            layer_name = layer_file[:-4]
            layer_image = cv2.imread("{}/{}".format(sample_dir, layer_file), cv2.IMREAD_COLOR)
            _, w, _ = layer_image.shape
            layer_image = cv2.resize(layer_image[:,3*w//4:], target_size)
            layer_mask = np.zeros(target_size)

            layer_mask_path = "{}/{}/{:0>2}.png".format(sample_dir, layer_name, object_id)
            if os.path.exists(layer_mask_path):
                layer_mask = cv2.imread(layer_mask_path, cv2.IMREAD_GRAYSCALE)
                layer_mask = cv2.resize(layer_mask, target_size)
            
            if frame_idx in prompt_idxs and np.sum(layer_mask) > 0:
                prompts_dct[frame_idx] = {}
                prompts_dct[frame_idx][object_id] = {}

                coord_positive, coord_negative = self.pg.get_prompt_points(layer_mask, self.ppp, self.ppn)

                prompts_dct[frame_idx][object_id]["points"] = coord_positive
                prompts_dct[frame_idx][object_id]["points"] += coord_negative

                prompts_dct[frame_idx][object_id]["labels"] = [1] * self.ppp + [0] * self.ppn

                # convert to correct numpy type
                prompts_dct[frame_idx][object_id]["points"] = np.array(prompts_dct[frame_idx][object_id]["points"], dtype=np.float32)
                prompts_dct[frame_idx][object_id]["labels"] = np.array(prompts_dct[frame_idx][object_id]["labels"], dtype=np.int32)

            image_seq.append(layer_image)
            mask_seq.append({object_id:layer_mask / 255})

        return image_seq, mask_seq, prompts_dct

    def load_sequence_faz(self, sample_name, selected_layer_sequence, prompt_idxs, object_id=0):
        sample_dir = "{}/{}".format(self.layer_root_dir, sample_name)

        target_size = (1024, 1024)

        image_seq, mask_seq = [], []

        prompts_dct = {}


        for frame_idx, layer_file in enumerate(selected_layer_sequence):
            layer_image_concat = cv2.imread("{}/{}".format(sample_dir, layer_file), cv2.IMREAD_GRAYSCALE)
            _, w = layer_image_concat.shape
     
            layer_image = cv2.resize(layer_image_concat[:, w//3:2*w//3], target_size)
            layer_mask = cv2.resize(layer_image_concat[:, 2*w//3:], target_size)

            if frame_idx in prompt_idxs and np.sum(layer_mask) > 0:
                prompts_dct[frame_idx] = {}
                prompts_dct[frame_idx][object_id] = {}

                coord_positive, coord_negative = self.pg.get_prompt_points(layer_mask, self.ppp, self.ppn)

                prompts_dct[frame_idx][object_id]["points"] = coord_positive
                prompts_dct[frame_idx][object_id]["points"] += coord_negative

                prompts_dct[frame_idx][object_id]["labels"] = [1] * self.ppp + [0] * self.ppn

                # convert to correct numpy type
                prompts_dct[frame_idx][object_id]["points"] = np.array(prompts_dct[frame_idx][object_id]["points"], dtype=np.float32)
                prompts_dct[frame_idx][object_id]["labels"] = np.array(prompts_dct[frame_idx][object_id]["labels"], dtype=np.int32)

            image_seq.append(self.to_3ch(layer_image))
            mask_seq.append({object_id:layer_mask / 255})

        return image_seq, mask_seq, prompts_dct


    def __len__(self):
        return len(self.batch_dct_lst)

    def __getitem__(self, index):
        return self.batch_dct_lst[index]

class OCTA_Dataset_SAM2_Projection(Dataset):
    def __init__(self, subset_name="Training", fov="3M", label_type="RV"):
        self.subset_name = subset_name
        self.label_type = label_type

        self.valid_batch_dct = {}

        self.data_dir = "datasets/OCTA500/ProjectionSamples"
    
        colors = [tuple(x) for x in np.load("color_list.npy")]
        self.colors_dct = dict(zip(colors, range(len(colors))))
        
        self.sample_names = sorted([x[:-4] for x in os.listdir(self.data_dir)])

        subset = set(list(range(10001, 10181))) if fov == "6M" else set(list(range(10301, 10441)))

        if subset_name=="Validation":
            subset = set(list(range(10181, 10201))) if fov == "6M" else set(list(range(10441, 10451)))
        elif subset_name=="Test":
            subset = set(list(range(10201, 10301))) if fov == "6M" else set(list(range(10451, 10501)))
        
        self.sample_names = [x for x in self.sample_names if int(x) in subset]

        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

        probability = 0.5
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=probability),
            alb.SafeRotate(limit=15, p=probability),
            alb.HorizontalFlip(p=probability),
            alb.AdvancedBlur(p=probability)
        ])
    def find_nearest_pixel(sefl, binary_image, input_coord):
        ones_coords = np.argwhere(binary_image == 1)
        distances = np.linalg.norm(ones_coords - input_coord, axis=1)
        nearest_index = np.argmin(distances)
        nearest_coord = ones_coords[nearest_index]
        return nearest_coord
    
    def load_sequence(self, sample_name):
        sample_image = cv2.imread("{}/{}.png".format(self.data_dir, sample_name), cv2.IMREAD_GRAYSCALE)
        _, w = sample_image.shape

        layer_1 = sample_image[:, :w//5]
        layer_2 = sample_image[:, w//5:2*w//5]
        layer_3 = sample_image[:, 2*w//5:3*w//5]

        image = np.array([layer_1, layer_2, layer_3]).transpose((1,2,0))

        mask = sample_image[:,-2*w//5:-w//5]
        if self.label_type == "FAZ": mask = sample_image[:,-w//5:]

        if self.subset_name == "Training":
            transformed = self.transform(**{"image": image, "mask": mask})
            image, mask = transformed["image"], transformed["mask"]
        
        object_id = 0
        layer_1, layer_2, layer_3 = image[:,:,0], image[:,:,1], image[:,:,2]
        layer_1, layer_2, layer_3 = map(self.to_3ch, [layer_1, layer_2, layer_3])

        image_seq = np.array([layer_1, layer_2, layer_3])
        mask_seq = [{object_id:mask / 255}] * 3

        prompts_dct = {}
        binary_image = mask.astype(np.uint8)
        binary_image[binary_image >= 1] = 1
        labeled_image, num_features = label(binary_image)
        centroids = center_of_mass(binary_image, labeled_image, range(1, num_features + 1))
        centroids = [self.find_nearest_pixel(binary_image, p) for p in centroids]
        prompt_points = [(y, x) for (x, y) in centroids]

        if self.label_type == "FAZ":
            eroded_img = binary_image - cv2.erode(binary_image, np.ones((7, 7), np.uint8), iterations=1)
            prompt_points += sample([[y, x] for x, y in np.argwhere(eroded_img == 1)], 3)

            
        for frame_idx in range(3):
            prompts_dct[frame_idx] = {}
            prompts_dct[frame_idx][object_id] = {}
    
            prompts_dct[frame_idx][object_id]["points"] = prompt_points
            prompts_dct[frame_idx][object_id]["labels"] = [1] * len(prompt_points)

            prompts_dct[frame_idx][object_id]["points"] = np.array(prompts_dct[frame_idx][object_id]["points"], dtype=np.float32)
            prompts_dct[frame_idx][object_id]["labels"] = np.array(prompts_dct[frame_idx][object_id]["labels"], dtype=np.int32)

        image_seq = np.array(image_seq)

        return image_seq, mask_seq, prompts_dct
 
    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, index):
        # prompts_lst -> coords(x, y), pos/neg, object_id: [x, y, 1/0, 0...n], four elements tuple
        sample_name = self.sample_names[index]

        if sample_name in self.valid_batch_dct and self.subset_name != "Training":
            return self.valid_batch_dct[sample_name]
        
        image_seq, mask_dct_seq, prompts_dct = self.load_sequence(sample_name)
        image_seq = np.array(image_seq).transpose((0,3,1,2))
        mask_seq = mask_dct_seq

        batch = {
            "sample_name": sample_name,
            "images": image_seq,
            "masks": mask_seq,
            "prompts": prompts_dct
        }

        if self.subset_name != "Training": self.valid_batch_dct[sample_name] = batch

        return batch       

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
        

