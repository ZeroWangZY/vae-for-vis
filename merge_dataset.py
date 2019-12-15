import random
import json

file_prefix = "real_";
IMAGE_DIM = 112;
limit_per_label = 120

path_for_read_of_scatters = "data/scatters_points_" + str(IMAGE_DIM) + ".json";
path_for_read_of_real_scatters = "data/" + file_prefix + "scatters_points_" + str(IMAGE_DIM) + ".json";

path_for_read_of_imgs = "data/scatters_" + str(IMAGE_DIM) + ".json";
path_for_read_of_real_imgs = "data/" + file_prefix + "scatters_" + str(IMAGE_DIM) + ".json";

path_for_read_of_labels = "data/scatters_labels_" + str(IMAGE_DIM) + ".json";
path_for_read_of_real_labels = "data/" + file_prefix + "scatters_labels_" + str(IMAGE_DIM) + ".json";

path_for_read_of_valid = "data/scatters_valid_labels_" + str(IMAGE_DIM) + ".json";
path_for_read_of_real_valid = "data/" + file_prefix + "scatters_valid_labels_" + str(IMAGE_DIM) + ".json";

path_for_write_of_merged_dataset = "rawdata/merged_" + str(IMAGE_DIM) + ".json";


def filter_func(arr, valid_set):
    ret = []
    length = len(valid_set)
    for i in range(length):
        ret.append(arr[valid_set[i]])
    return ret

def compose_valid_dataset(scatters, imgs, labels, tgt_label, replace_label):
    ret_dataset = [];
    length = len(labels);

    for i in range(length):
        if labels[i]["label"] == tgt_label:
            ret_dataset.append({
              "img": imgs[i],
              # "scatter": scatters[i],
              "label": replace_label
            })

    return ret_dataset;

with open(path_for_read_of_scatters, 'r') as load_f:
    scatters = json.load(load_f)

with open(path_for_read_of_real_scatters, 'r') as load_f:
    real_scatters = json.load(load_f)

with open(path_for_read_of_imgs, 'r') as load_f:
    imgs = json.load(load_f)

with open(path_for_read_of_real_imgs, 'r') as load_f:
    real_imgs = json.load(load_f)

with open(path_for_read_of_labels, 'r') as load_f:
    labels = json.load(load_f)

with open(path_for_read_of_real_labels, 'r') as load_f:
    real_labels = json.load(load_f)

with open(path_for_read_of_valid, 'r') as load_f:
    valid_set = json.load(load_f)

with open(path_for_read_of_real_valid, 'r') as load_f:
    real_valid_set = json.load(load_f)

# imgs = filter_func(imgs, valid_set)
# scatters = filter_func(scatters, valid_set)
# labels = filter_func(labels, valid_set)

# real_imgs = filter_func(real_imgs, real_valid_set)
# real_scatters = filter_func(real_scatters, real_valid_set)
# real_labels = filter_func(real_labels, real_valid_set)

ret = []

# 0和5数据量不够，不要了
label_mapping = [0, 0, 1, 2, 3, 5, 4, 5, 6]

for i in range(9):
    print("round " + str(i))

    if i == 0 or i == 5:
        continue

    replace_label = label_mapping[i];

    valid_dataset = compose_valid_dataset(scatters, imgs, labels, i, replace_label);
    real_valid_dataset = compose_valid_dataset(real_scatters, real_imgs, real_labels, i, replace_label);

    merged_dataset = valid_dataset + real_valid_dataset
    random.shuffle(merged_dataset)
    merged_dataset.sort(reverse = True, key = lambda item: item["label"])
    merged_dataset = merged_dataset[0: limit_per_label]

    ret += merged_dataset

print("length of dataset: ", len(ret))

with open(path_for_write_of_merged_dataset, "w") as f:
    json.dump(ret, f)