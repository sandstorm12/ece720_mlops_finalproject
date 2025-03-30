from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Tuple
import torch
import numpy as np
import cv2

def generate_batch_heatmaps(
    model: torch.nn.Module,
    target_layers: List,
    batch: dict,
    device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate GradCAM heatmaps and overlay images for a batch of samples.

    :param model: Trained model (e.g., ResNet50)
    :param target_layers: List of layers for GradCAM (usually [model.layer4[-1]])
    :param batch: Dict from DataLoader (should contain "image", "id", "og_image")
    :param device: torch.device
    :return: Tuple of:
        - List of raw heatmaps [H, W] as NumPy arrays
        - List of heatmaps overlaid on original images [H, W, 3] as uint8 arrays
    """
    model.eval().to(device)

    images = batch["image"].to(device)         # [B, 3, 224, 224]
    og_images = batch["og_image"]              # List or tensor: (B, H, W, 3)
    class_ids = batch["id"]                    # [B]

    heatmaps = []
    overlays = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for i in range(images.shape[0]):
            input_tensor = images[i].unsqueeze(0)  # [1, 3, 224, 224]
            targets = [ClassifierOutputTarget(class_ids[i].item())]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)
            heatmaps.append(grayscale_cam)

            # Make sure original image is in numpy format and [0, 1] range
            if torch.is_tensor(og_images[i]):
                og_img = og_images[i].cpu().numpy()
            else:
                og_img = og_images[i]

            # Ensure correct type/shape and normalize to [0, 1]
            if og_img.max() > 1.0:
                og_img = og_img.astype(np.float32) / 255.0

            cam_image = show_cam_on_image(og_img, grayscale_cam, use_rgb=True)
            overlays.append(cam_image)

    return heatmaps, overlays



import matplotlib.pyplot as plt

def visualize_heatmap_seg_overlay(
    batch: dict,
    heatmaps: List[np.ndarray],
    overlays: List[np.ndarray],
    max_images: int = 3
):
    """
    Plot original images, segmentation masks, and GradCAM overlays in a 3x3 grid.

    :param batch: A batch from the DataLoader (must contain "og_image", "seg", "name")
    :param heatmaps: List of heatmaps (H, W)
    :param overlays: List of GradCAM overlays (H, W, 3)
    :param max_images: Number of images to visualize (default 3)
    """
    max_images = min(max_images, len(overlays))

    fig, axes = plt.subplots(3, max_images, figsize=(5 * max_images, 12))

    for i in range(max_images):
        # Original image
        og_img = batch["og_image"][i]
        if torch.is_tensor(og_img):
            og_img = og_img.cpu().numpy()
        og_img = og_img.astype(np.uint8)

        # Segmentation mask
        seg_mask = batch["seg"][i]
        if torch.is_tensor(seg_mask):
            seg_mask = seg_mask.cpu().numpy()

        # Class name
        class_name = batch["name"][i]

        # --- Row 1: Original image ---
        axes[0, i].imshow(og_img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{class_name}", fontsize=14)
        if i == 0:
            axes[0, i].set_ylabel("Original Image", fontsize=12)

        # --- Row 2: Segmentation ---
        axes[1, i].imshow(seg_mask, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Segmentation", fontsize=12)

        # --- Row 3: GradCAM ---
        axes[2, i].imshow(overlays[i])
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("GradCAM", fontsize=12)

    plt.tight_layout()
    plt.show()




def threshold_batch_heatmaps(
    batch: dict,
    heatmaps: List[np.ndarray],
    percentile: float = 80.0,
    min_region_area: int = 50,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Threshold GradCAM heatmaps for a batch and return cleaned binary masks + overlayed images.

    :param batch: Batch from DataLoader (must contain "og_image")
    :param heatmaps: List of 2D GradCAM heatmaps (one per sample)
    :param percentile: Threshold top X% (default = top 20%)
    :param min_region_area: Minimum area (in px) to keep connected regions
    :param blur_kernel_size: Kernel size for Gaussian blur (should be odd)
    :param blur_sigma: Sigma for Gaussian blur
    :return: Tuple of (binary_masks, overlay_images)
    """
    binary_masks = []
    overlays = []

    for i in range(len(heatmaps)):
        heatmap = heatmaps[i]

        # Step 1: Smooth the heatmap
        heatmap_smooth = cv2.GaussianBlur(heatmap, (blur_kernel_size, blur_kernel_size), sigmaX=blur_sigma)

        # Step 2: Percentile threshold
        thresh = np.percentile(heatmap_smooth, percentile)
        binary_mask = (heatmap_smooth >= thresh).astype(np.uint8)

        # Step 3: Remove small scattered regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned = np.zeros_like(binary_mask)
        for j in range(1, num_labels):  # skip background
            if stats[j, cv2.CC_STAT_AREA] >= min_region_area:
                cleaned[labels == j] = 1

        binary_masks.append(cleaned)

        # Step 4: Overlay cleaned mask on image
        og_image = batch["og_image"][i]
        if torch.is_tensor(og_image):
            og_image = og_image.cpu().numpy()
        if og_image.dtype != np.uint8:
            og_image = (og_image * 255).astype(np.uint8)

        image_float = og_image.astype(np.float32) / 255.0
        overlay = show_cam_on_image(image_float, cleaned.astype(np.float32), use_rgb=True)
        overlays.append(overlay)

    return binary_masks, overlays


from typing import List
import numpy as np

def compute_batch_iou(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray]
) -> List[float]:
    """
    Compute IoU between batches of ground truth and predicted segmentation masks.

    :param gt_masks: List of GT masks (2D binary arrays)
    :param pred_masks: List of predicted masks (2D binary arrays)
    :return: List of IoU scores per sample
    """
    assert len(gt_masks) == len(pred_masks), "GT and Pred mask batches must match in length"

    ious = []

    for gt, pred in zip(gt_masks, pred_masks):
        gt_bin = gt.astype(bool)
        pred_bin = pred.astype(bool)

        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()

        if union == 0:
            iou = float('nan')  # undefined if both masks are empty
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def containment_score_batch_np(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray
) -> np.ndarray:
    """
    Compute how much of the smaller mask is contained within the larger one (per image).

    :param gt_masks: np.ndarray of shape (B, H, W), binary ground truth masks
    :param pred_masks: np.ndarray of shape (B, H, W), binary predicted masks
    :return: np.ndarray of shape (B,) with containment scores (float)
    """
    assert gt_masks.shape == pred_masks.shape, "Shape mismatch between GT and Pred masks"
    
    scores = []

    for i in range(gt_masks.shape[0]):
        gt = gt_masks[i].astype(bool)
        pred = pred_masks[i].astype(bool)

        # Identify smaller and larger mask
        if gt.sum() < pred.sum():
            small, large = gt, pred
        else:
            small, large = pred, gt

        intersection = np.logical_and(small, large).sum()
        small_area = small.sum()

        if small_area == 0:
            scores.append(float('nan'))
        else:
            scores.append(intersection / small_area)

    return np.array(scores)