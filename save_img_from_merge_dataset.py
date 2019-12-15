import json
import imageio, os
import numpy as np

IMAGE_DIM = 112;

path_for_read_of_merge_dataset = "rawdata/merged_" + str(IMAGE_DIM) + ".json";
path_for_write_of_img2label = "data/img2label_" + str(IMAGE_DIM) + ".json";

def gen_scatters_image(scatters_data, path):
    figure = np.zeros((IMAGE_DIM, IMAGE_DIM))
    figure[0: IMAGE_DIM, 0: IMAGE_DIM] = scatters_data
    imageio.imwrite(path, figure * 255)

with open(path_for_read_of_merge_dataset, 'r') as load_f:
    ret = json.load(load_f)

length_of_dataset = len(ret)

img2label = {};

indice_mapping = ["convex", "skinny", "stringy", "straight", "skew", "clumpy", "striated"]

for i in range(length_of_dataset):
    print("saving img: {} / {}".format(i, length_of_dataset))
    data = ret[i]
    label = data["label"]
    img_path = "data/png/" + indice_mapping[label] + "_" + str(i) + ".png"
    gen_scatters_image(data["img"], img_path)
    img2label[img_path] = label

with open(path_for_write_of_img2label, "w") as f:
    json.dump(img2label, f)