import cv2
import yaml
import torch
import numpy as np
import gradio as gr

from torchvision import models

from gradcam_tools import generate_heatmap, get_superimposed_image
from utils.shap_explainer_gradio import SHAPImageExplainer


SAMPLES = [
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n01443537/ILSVRC2012_val_00004677.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n02123597/ILSVRC2012_val_00017692.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n02483362/ILSVRC2012_val_00008017.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n02783161/ILSVRC2012_val_00001098.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n02992529/ILSVRC2012_val_00008541.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n03201208/ILSVRC2012_val_00015035.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n03452741/ILSVRC2012_val_00018721.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n04026417/ILSVRC2012_val_00010949.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n04447861/ILSVRC2012_val_00002574.JPEG", "Grad-Cam", 0],
    ["/home/hamid/Documents/uofa/MLOPs/ece720_mlops_finalproject/imagenet_val_s/ImageNetS50/validation/n06794110/ILSVRC2012_val_00001518.JPEG", "Grad-Cam", 0],
]

SAMPLES_LABEL = [
    "n01443537",
    "n02123597",
    "n02483362",
    "n02783161",
    "n02992529",
    "n03201208",
    "n03452741",
    "n04026417",
    "n04447861",
    "n06794110",
]


def _load_model():
    model = models.resnet50(pretrained=True)
    model.eval()

    return model


def _load_shap_explainer():
    explainer = SHAPImageExplainer()

    return explainer


def _load_labels(path):
    with open(path, 'r') as stream:
        raw = yaml.safe_load(stream)

    dir_2_id = {}
    id_2_name = {}

    for key in raw.keys():
        id = int(key)
        dir_2_id[raw[key][0]] = int(key)
        id_2_name[id] = raw[key][1]

    return dir_2_id, id_2_name


def _preprocess_img(image):
    image_resized = cv2.resize(image, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    image_tensor = torch.from_numpy(image_cropped).float()
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor /= 255.0

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]

    return image_tensor


def _get_shape_explaination(img):
    image_resized = cv2.resize(img, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    explaination_img, heatmap, predicted_class = explainer_shap.explain(image_cropped)

    if len(heatmap.shape) == 3:  # (height, width, channels)
        heatmap = np.mean(heatmap, axis=2)  # Average over channels, shape: (height, width)

    # Optional: Normalize the heatmap to [0, 1] for use as a mask
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return explaination_img, heatmap, predicted_class


def _get_grad_cam_explaination(img, pred_id):
    target_layers = [model.layer4[-1]]
    heatmap = generate_heatmap(model, target_layers, img, pred_id)
    visual = get_superimposed_image(img, heatmap)

    return heatmap, visual


def _get_class(img):
    image_resized = cv2.resize(img, (256, 256))
    h, w, _ = image_resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    image_cropped = image_resized[top:top+224, left:left+224]

    image_tensor = _preprocess_img(image_cropped)
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))

    _, predicted_idx = torch.max(output, 1)
    predicted_class_name = id_2_name[predicted_idx.item()]

    return predicted_class_name


def process_image(img, model_choice, blur_level, blur_choice):
    global previous_img
    global heatmap_cache
    global previous_method

    same_img = previous_img is not None and \
        (img.shape != previous_img.shape or (img != previous_img).all())
    cache = heatmap_cache is None or previous_method != model_choice
    
    if same_img or cache:
        image_tensor = _preprocess_img(img)
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))

        _, predicted_idx = torch.max(output, 1)
        true_class_name = _get_class(img)

        if model_choice == "Grad-Cam":
            heatmap, visual = _get_grad_cam_explaination(img, predicted_idx)            
            heatmap_cache = heatmap.copy()
            heatmap *= 255.0
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB).astype(np.uint8)

            predicted_class_name = id_2_name[predicted_idx.item()]

            img_grad = cv2.hconcat([visual, heatmap])

            return_img = img_grad            
        elif model_choice == "SHAP":
            img_shap, heatmap, predicted_class = \
                _get_shape_explaination(img)
            
            heatmap_cache = heatmap.copy()

            predicted_class_name = id_2_name[predicted_class.item()]

            return_img = img_shap
        
        previous_img = img
        previous_method = model_choice
        
        return return_img, predicted_class_name, true_class_name
    else:
        true_class_name = _get_class(img)

        image_resized = cv2.resize(img, (256, 256))
        h, w, _ = image_resized.shape
        top = (h - 224) // 2
        left = (w - 224) // 2
        image_cropped = image_resized[top:top+224, left:left+224]

        blurred_image = cv2.GaussianBlur(image_cropped, (99, 99), sigmaX=21, sigmaY=21)
        blurred_image = cv2.addWeighted(blurred_image, blur_level / 100,
                                        image_cropped, 1 - blur_level / 100, 0)
        
        if blur_choice == "Positive":
            image_cropped[heatmap_cache > heatmap_cache.mean()] = \
                blurred_image[heatmap_cache > heatmap_cache.mean()]
        elif blur_choice == "Negative":
            image_cropped[heatmap_cache < heatmap_cache.mean()] = \
                blurred_image[heatmap_cache < heatmap_cache.mean()]

        image_tensor = _preprocess_img(image_cropped)
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))

        _, predicted_idx = torch.max(output, 1)

        predicted_class_name = id_2_name[predicted_idx.item()]

        return image_cropped, predicted_class_name, true_class_name


def _start():
    img_in = gr.Image(type="numpy")
    model_choice = gr.Radio(
        choices=["Grad-Cam", "SHAP"],
        label="Select Model(s)",
        value="Grad-Cam",
    )
    blur_choice = gr.Radio(
        choices=["Positive", "Negative"],
        label="Blur region",
        value="Positive",
    )
    blur_level = gr.Slider(0, 100, step=1, label="Blur Level")
    inputs = [img_in, model_choice, blur_level, blur_choice]
    
    img_out = gr.Image(type="numpy")
    label_pred = gr.Textbox(label="Predicted Label")
    label_true = gr.Textbox(label="True Label")
    outputs = [img_out, label_pred, label_true]

    iface = gr.Interface(
        fn=process_image,
        inputs=inputs,
        outputs=outputs,
        title="Image Classification with Heatmap",
        examples=SAMPLES,
    )

    iface.launch()


if __name__ == "__main__":
    previous_img = None
    previous_method = None
    heatmap_cache = None

    explainer_shap = _load_shap_explainer()

    dir_2_id, id_2_name = _load_labels("labels.yaml")

    model = _load_model()

    _start()
