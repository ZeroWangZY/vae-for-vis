import numpy as np
import imageio, os
import json
import shutil
from image_drawer import gen_scatters_image

img_dim = 112
tgt_dir = "image/image_by_label/"

# 默认从0号开始
def find_num_of_class(scatters_label):
    labels = []
    for _, label_info in enumerate(scatters_label):
        labels.append(int(label_info["label"]))
    return max(labels) + 1


def split_images_by_label(scatters_data, scatters_label):
    num_of_class = find_num_of_class(scatters_label)
    images_by_label = [[] for i in range(num_of_class)]
    for index, label_info in enumerate(scatters_label):
        label = int(label_info["label"])
        image = scatters_data[index]
        images_by_label[label].append(image)
    return images_by_label

def get_indices_name(scatters_label):
    return list(scatters_label[0]["indices"].keys());


scatters_data = []
scatters_label = []

with open("data/scatters_" + str(img_dim) + ".json", 'r') as load_f:
    scatters_data = json.load(load_f)

with open("data/scatters_labels_" + str(img_dim) + ".json", 'r') as load_f:
    scatters_label = json.load(load_f)

indices_name = get_indices_name(scatters_label)

images_by_label = split_images_by_label(scatters_data, scatters_label)

for label, images in enumerate(images_by_label):
    path = tgt_dir + "labelled_scatters_" + str(img_dim) + "_" + str(label) + "_" + indices_name[label] + ".png"
    gen_scatters_image(images, path)
