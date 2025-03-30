import cv2
import torch
import numpy as np

from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def _load_model():
    model = models.resnet50(pretrained=True)
    model.eval()

    return model


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


def generate_heatmap(model, target_layers, img, id):
    image_resized = cv2.resize(img, (256, 256))
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

    input = image_tensor.unsqueeze(0)
    targets = [ClassifierOutputTarget(id)]

    with GradCAM(model, target_layers=target_layers) as gradcam:
        grayscale_cam = gradcam(input_tensor=input, targets=targets)

    heatmap = grayscale_cam[0, :]
    
    return heatmap

def get_superimposed_image(img, heatmap):
    image_resized = cv2.resize(img, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    image = image_cropped / 255.0
    visual = show_cam_on_image(image, heatmap)
    
    return visual


if __name__ == "__main__":
    model = _load_model()
    target_layers = [model.layer4[-1]]

    img_path = "/home/hamid/Documents/uofa/MLOPs/imagenet_val_s/ImageNetS50/validation/n01443537/ILSVRC2012_val_00004677.JPEG"
    img_sample = cv2.imread(img_path)

    image_rgb = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)

    # image_tensor = _preprocess_img(image_rgb)
    # with torch.no_grad():
    #     output = model(image_tensor.unsqueeze(0))

    # _, predicted_idx = torch.max(output, 1)
    # print(predicted_idx)

    heatmap = generate_heatmap(model, target_layers, image_rgb, 971)
    visual = get_superimposed_image(image_rgb, heatmap)

    cv2.imshow("heatmap", heatmap)
    cv2.imshow("visual", visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()