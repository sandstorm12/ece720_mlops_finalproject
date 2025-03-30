import os
import cv2
import glob
import yaml

import random
import requests
from pathlib import Path
from typing import List, Tuple
import cv2

import numpy as np

imagenet_s_50 = [
    "n01443537","n01491361","n01531178","n01644373","n02104029","n02119022","n02123597","n02133161","n02165456","n02281406","n02325366",
    "n02342885","n02396427","n02483362","n02504458","n02510455","n02690373","n02747177","n02783161","n02814533","n02859443","n02917067",
    "n02992529", "n03014705","n03047690","n03095699","n03197337","n03201208","n03445777","n03452741","n03584829","n03630383","n03775546",
    "n03791053","n03874599","n03891251","n04026417","n04335435","n04380533","n04404412","n04447861","n04507155","n04522168","n04557648",
    "n04562935","n04612504","n06794110","n07749582","n07831146","n12998815",
]




class Dataset:

    def __init__(self, dataset_path: Path, labels_path: Path, indices: List[int], image_size: Tuple[int]):
        """
        Define the image paths from the dataset.

        :param dataset_path: The dataset folder. Should contain the `validation` and `validation-segmentation` directories.
        :param labels_path: The `labels.yaml` file
        :param indices: The indices of the images in the dataset. (List containing the subdirectories)
        :param image_size: The size of the image, used for resizing and should match model input size.
        """
        self.dir2id, self.id2dir = self._load_labels(labels_path)
        self.indices = indices
        self.labels = self._load_class_labels()
        self.image_size = image_size

        # Construct a list that stores the paths of the regular image, the segmented image, id, and the name.
        self.img_path = dataset_path / "validation"
        self.seg_img_path = dataset_path / "validation-segmentation"
        self.images = self._load_image_info(self.img_path, self.seg_img_path, self.dir2id, self.id2dir)

    def sample(self, count: int):
        """
        Generate a sample from the dataset. The returned list contains dictionaries in the following format:

        ```json
        {
                "og_image": <np.ndarray>,
                "image": <torch.Tensor>,
                "seg": <np.ndarray>,
                "id": int,
                "name": str,
        }
        ```

        :param count: The number of samples to generate.
        :return: A list of the samples.
        """
        selection = random.sample(self.images, count)
        sample = []

        for i in range(len(selection)):  # type: ignore
            image_path, seg_path, id, name, dir_name = selection[i]["image_path"], selection[i]["seg_path"], selection[i]["id"], selection[i]["name"], selection[i]["dir_name"]

            og_image = cv2.imread(str(image_path))
            image = self._preprocess_img(og_image)
            seg = cv2.imread(str(seg_path))
            seg = self._get_binary_mask(seg, dir_name)

            # Resize both images to be the same as ResNet
            og_image = cv2.resize(og_image, self.image_size)
            seg = cv2.resize(seg, self.image_size)

            sample.append({
                "og_image": og_image,
                "image": image,
                "seg": seg,
                "id": id,
                "name": name,
            })

        return sample

    def logits2label(self, logits):
        """
        Convert the logits generated from the model to the output label name. This only works for one label.

        :param logits: The model generated logits.
        :return: The output label name.
        """
        _, predicted_idx = torch.max(logits, 1)
        predicted_label = self.labels[str(predicted_idx.item())][1]
        return predicted_label

    def _load_labels(self, labels_path: Path):
        with open(labels_path, 'r') as stream:
            raw = yaml.safe_load(stream)

        dir_2_id = {}
        id_2_name = {}

        for key in raw.keys():
            id = int(key)
            dir_2_id[raw[key][0]] = int(key)
            id_2_name[id] = raw[key][1]

        return dir_2_id, id_2_name

    def _load_image_info(self, path_img, path_seg, dir_2_id, id_2_name):
        image_paths = glob.glob(os.path.join(path_img, '**', '*.JPEG'))

        images_info = []
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_dir = os.path.basename(os.path.dirname(image_path))

            seg_path = os.path.join(path_seg, image_dir, image_name)
            seg_path = seg_path.replace("JPEG", "png")

            if not os.path.exists(seg_path):
                continue

            images_info.append(
                {
                    "image_path": image_path,
                    "seg_path": seg_path,
                    "dir_name": image_dir,
                    "id": dir_2_id[image_dir],
                    "name": id_2_name[dir_2_id[image_dir]]
                }
            )

        return images_info

    def _get_binary_mask(self, seg_mask, dirname):
        id = self.indices.index(dirname) + 1

        red = id % 256
        green = (id // 256)

        mask_pixels = (seg_mask == [0, green, red]).all(axis=-1)

        seg_mask[mask_pixels] = [255, 255, 255]
        seg_mask[~mask_pixels] = [0, 0, 0]

        seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)

        return seg_mask

    def _preprocess_img(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image_resized = cv2.resize(image_rgb, (256, 256))
        # h, w, _ = image_resized.shape
        # top = (h - 224) // 2
        # left = (w - 224) // 2
        # image_cropped = image_resized[top:top+224, left:left+224]

        # image_tensor = torch.from_numpy(image_cropped).float()
        # image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
        # image_tensor /= 255.0  # Normalize pixel values to [0, 1]

        # mean = torch.tensor([0.485, 0.456, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        # image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]

        return image_tensor

    def _load_class_labels(self):
        imagenet_class_index_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        response = requests.get(imagenet_class_index_url)
        class_idx = response.json()

        return class_idx
    


    def _get_sample_data(self,count=64):

        samples=self.sample(count=count)
        X=np.array([samples[i]['og_image'] for i in range(count)])
        y_id=np.array([samples[i]['id'] for i in range(count)])
        y_name=np.array([samples[i]['name'] for i in range(count)])

        return X,y_id,y_name

