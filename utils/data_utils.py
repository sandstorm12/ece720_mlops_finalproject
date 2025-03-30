import os
import cv2
import glob
import yaml


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


