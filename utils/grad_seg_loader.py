import os
import cv2
import glob
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import torchvision

import matplotlib.pyplot as plt 

# ---------- 50-class subset of ImageNet ----------
imagenet_s_50 = [
    "n01443537","n01491361","n01531178","n01644373","n02104029","n02119022","n02123597","n02133161","n02165456","n02281406","n02325366",
    "n02342885","n02396427","n02483362","n02504458","n02510455","n02690373","n02747177","n02783161","n02814533","n02859443","n02917067",
    "n02992529", "n03014705","n03047690","n03095699","n03197337","n03201208","n03445777","n03452741","n03584829","n03630383","n03775546",
    "n03791053","n03874599","n03891251","n04026417","n04335435","n04380533","n04404412","n04447861","n04507155","n04522168","n04557648",
    "n04562935","n04612504","n06794110","n07749582","n07831146","n12998815",
]

# ---------- Dataset Class ----------
class ImageNetS50DatasetDict(Dataset):
    def __init__(self, dataset_path: Path, labels_path: Path, class_dirs: List[str], image_size: Tuple[int], transform=None):
        """
        Dataset for ImageNet-S50 with segmentation support.

        :param dataset_path: Path to root dataset (expects 'validation' and 'validation-segmentation')
        :param labels_path: Path to labels.yaml
        :param class_dirs: List of 50 ImageNet-S class directories
        :param image_size: Tuple of (W, H)
        :param transform: Optional torchvision transform for image
        """
        self.image_size = image_size
        self.transform = transform

        self.dir2id, self.id2name = self._load_labels(labels_path)
        self.valid_class_dirs = class_dirs

        self.image_paths = []
        self.seg_paths = []
        self.labels = []
        self.class_names = []

        for class_dir in self.valid_class_dirs:
            val_path = dataset_path / "validation" / class_dir
            seg_path = dataset_path / "validation-segmentation" / class_dir

            if val_path.exists():
                images = glob.glob(str(val_path / "*.JPEG"))
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    mask_path = str(seg_path / img_name).replace(".JPEG", ".png")

                    if not os.path.exists(mask_path):
                        continue  # skip if mask missing

                    self.image_paths.append(img_path)
                    self.seg_paths.append(mask_path)
                    self.labels.append(self.dir2id[class_dir])
                    self.class_names.append(self.id2name[self.dir2id[class_dir]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.seg_paths[idx]
        label_id = self.labels[idx]
        label_name = self.class_names[idx]

        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.image_size)
        og_image = img_resized.copy()

        # Load and resize segmentation mask
        seg_mask = cv2.imread(seg_path)
        seg_binary = self._get_binary_mask(seg_mask, img_path)
        seg_binary = cv2.resize(seg_binary, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Transform image
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "og_image": og_image,
            "seg": seg_binary,
            "id": label_id,
            "name": label_name,
            "path": img_path,
        }

    def _get_binary_mask(self, seg_mask, img_path: str):
        """
        Convert the segmentation RGB mask into binary mask using class ID.
        """
        class_dir = os.path.basename(os.path.dirname(img_path))
        class_idx = self.valid_class_dirs.index(class_dir) + 1

        red = class_idx % 256
        green = (class_idx // 256)

        # Match (0, green, red)
        mask_pixels = (seg_mask == [0, green, red]).all(axis=-1)

        binary_mask = np.zeros(seg_mask.shape[:2], dtype=np.uint8)
        binary_mask[mask_pixels] = 255  # Foreground is white (target class)

        return binary_mask

    def _load_labels(self, labels_path: Path):
        with open(labels_path, 'r') as stream:
            raw = yaml.safe_load(stream)

        dir2id = {}
        id2name = {}
        for key in raw:
            id = int(key)
            dir2id[raw[key][0]] = id
            id2name[id] = raw[key][1]

        return dir2id, id2name


# ---------- Optional Preprocessing Transform ----------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Converts HWC [0–255] to CHW [0.0–1.0]
    torchvision.transforms.Normalize(mean=mean, std=std),
])


from torchvision.models import resnet50, ResNet50_Weights

def load_model(device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    return model
