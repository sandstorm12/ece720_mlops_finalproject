import os
import cv2
import glob
import yaml
import numpy as np

from mapper import imagenet_s_50


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


def _load_image_info(path_img, path_seg, dir_2_id, id_2_name):
    image_paths = glob.glob(os.path.join(path_img, '**', '*.JPEG'))
    seg_paths = glob.glob(os.path.join(path_seg, '**', '*.JPEG'))

    images_info = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_dir = os.path.basename(os.path.dirname(image_path))
        
        seg_path = os.path.join(path_seg, image_dir, image_name)
        seg_path = seg_path.replace("JPEG", "png")
        if os.path.exists(seg_path) == False:
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


def _display_sample(images_info):
    for i in range(len(images_info)):
        image_path = images_info[i]["image_path"]
        seg_path = images_info[i]["seg_path"]
        id = images_info[i]["id"]
        name = images_info[i]["name"]
        dir_name = images_info[i]["dir_name"]

        image = cv2.imread(image_path)
        
        seg = cv2.imread(seg_path)
        seg = _get_binary_mask(seg, dir_name)

        cv2.imshow(f"image: {id}: {name}", image)
        cv2.imshow(f"seg: {id}: {name}", seg)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()


def _get_binary_mask(seg_mask, dirname):
    id = imagenet_s_50.index(dirname) + 1
    
    red = id % 256
    green = (id // 256)

    mask_pixels = (seg_mask == [0, green, red]).all(axis=-1)

    seg_mask[mask_pixels] = [255, 255, 255]
    seg_mask[~mask_pixels] = [0, 0, 0]

    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    
    return seg_mask


if __name__ == "__main__":
    dir_2_id, id_2_name = _load_labels("labels.yaml")
    
    images_info = _load_image_info(
        path_img="/home/hamid/Downloads/imagenet/imagenet_val_s/ImageNetS50/validation",
        path_seg="/home/hamid/Downloads/imagenet/imagenet_val_s/ImageNetS50/validation-segmentation",
        dir_2_id=dir_2_id,
        id_2_name=id_2_name,
    )

    _display_sample(images_info)
