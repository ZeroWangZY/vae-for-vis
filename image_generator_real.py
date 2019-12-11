import csv
import json
import numpy as np
import math
from collections import defaultdict

IMAGE_DIM = 112
MARGIN = 3

csv_filename = "rawdata/communites_and_crime_dataset.csv"

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def isItemValid(x, y):
    return (not math.isnan(x)) and (not math.isnan(y))

def gen_image(arr1, arr2):

    # init image
    result = [[0 for col in range(IMAGE_DIM)] for row in range(IMAGE_DIM)]

    scatter = []

    min1 = min(arr1)
    max1 = max(arr1)
    min2 = min(arr2)
    max2 = max(arr2)
    minP = MARGIN
    maxP = IMAGE_DIM - MARGIN
    for i in range(len(arr1)):
        x_val = arr1[i]
        y_val = arr2[i]
        if not isItemValid(x_val, y_val):
            continue
        try:
            x = round(minP + (maxP - minP) * normalize(arr1[i], min1, max1))
            y = round(minP + (maxP - minP) * normalize(arr2[i], min2, max2))
            scatter.append({
              'x': x_val,
              'y': y_val
            })
            for j in range(x - 2, x + 3):
                for k in range(y - 2, y + 3):
                    result[j][k] += 1
        except (ValueError, ZeroDivisionError):
            pass
        result = np.array(result)
    if np.max(result) <= 0:
        return None
    result = result / np.max(result)
    return result.tolist(), scatter


# 散点数据
scatters = []

# 对应的图像数据
images = []

csv_dict = defaultdict(list)
with open(csv_filename, mode = 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    csv_dict = {elem: [] for elem in csv_reader.fieldnames}
    for row in csv_reader:
        for key in csv_dict.keys():
            csv_dict[key].append(row[key])

keys = list(csv_dict.keys())
for i, key1 in enumerate(keys):
    for j, key2 in enumerate(keys):
        print("round: {}, {} / {}, {}".format(i, j, len(keys), len(keys)))
        if key1 != key2:
            col1 = list(map(float, csv_dict[key1]))
            col2 = list(map(float, csv_dict[key2]))
            d = gen_image(col1, col2)
            image, scatter = d
            images.append(image)
            scatters.append(scatter)

with open("data/real_scatters_" + str(IMAGE_DIM) + ".json", "w") as f:
    json.dump(images, f)

with open("data/real_scatters_points_" + str(IMAGE_DIM) + ".json", "w") as f:
    # json.dump(scatters, f, separators=(',', ':'), indent=4)
    json.dump(scatters, f)
