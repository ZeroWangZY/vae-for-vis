import json
import pickle
import imageio, os
import numpy as np

IMAGE_DIM = 112

path_for_read_of_merge_dataset = "rawdata/merged_" + str(IMAGE_DIM) + ".json"
path_for_write_of_img2label = "data/img2label"
path_for_write_of_label2class = "data/label2class"
path_for_write_of_img2multi_label = "data/img2multi_label"

def gen_scatters_image(scatters_data, path):
    figure = np.zeros((IMAGE_DIM, IMAGE_DIM))
    figure[0: IMAGE_DIM, 0: IMAGE_DIM] = scatters_data
    imageio.imwrite(path, figure * 255)

with open(path_for_read_of_merge_dataset, 'r') as load_f:
    ret = json.load(load_f)

length_of_dataset = len(ret)

img2label = {}

img2multi_label = {}

indice_mapping = ["convex", "skinny", "stringy", "straight", "skew", "clumpy", "striated"]

for i in range(length_of_dataset):
    print("saving img: {} / {}".format(i, length_of_dataset))
    data = ret[i]
    label = data["label"]
    indices = data["indices"]
    img_name = str(i) + ".png"
    img_path = "data/png/" + str(i) + ".png"
    gen_scatters_image(data["img"], img_path)
    img2label[img_name] = label
    img2multi_label[img_name] = indices

label2class = {}
for i in range(len(indice_mapping)):
    label2class[str(i)] = indice_mapping[i]

with open(path_for_write_of_img2label + ".pkl", 'wb') as f:
    pickle.dump(img2label, f, pickle.HIGHEST_PROTOCOL)

with open(path_for_write_of_img2label + ".json", "w") as f:
    json.dump(img2label, f)

with open(path_for_write_of_img2multi_label + ".pkl", 'wb') as f:
    pickle.dump(img2multi_label, f, pickle.HIGHEST_PROTOCOL)

with open(path_for_write_of_img2multi_label + ".json", "w") as f:
    json.dump(img2multi_label, f)

with open(path_for_write_of_label2class + ".pkl", 'wb') as f:
    pickle.dump(label2class, f, pickle.HIGHEST_PROTOCOL)

with open(path_for_write_of_label2class + ".json", "w") as f:
    json.dump(label2class, f)