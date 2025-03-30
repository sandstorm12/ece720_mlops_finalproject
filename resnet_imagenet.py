import cv2
import torch
import requests
import numpy as np

import torchvision.transforms as transforms

from torchvision import models


def _load_model():
    model = models.resnet50(pretrained=True)
    model.eval()

    return model


def _load_image(path):
    image = cv2.imread(path)

    return image


def _preprocess_img(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    image_tensor = torch.from_numpy(image_cropped).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
    image_tensor /= 255.0  # Normalize pixel values to [0, 1]

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]

    return image_tensor


def _load_labels():
    imagenet_class_index_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    response = requests.get(imagenet_class_index_url)
    class_idx = response.json()

    return class_idx


def _predict_class(labels):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))

    _, predicted_idx = torch.max(output, 1)

    predicted_label = labels[str(predicted_idx.item())][1]

    return predicted_label


if __name__ == "__main__":
    model = _load_model()

    image_path = "/home/hamid/Downloads/imagenet/imagenet_val_s/ImageNetS50/validation/n03791053/ILSVRC2012_val_00011514.JPEG"
    image = _load_image(image_path)
    
    image_tensor = _preprocess_img(image)

    labels = _load_labels()

    predicted_label = _predict_class(labels)

    print(f"Predicted class: {predicted_label}")

    
