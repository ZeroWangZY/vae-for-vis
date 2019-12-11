from pydataset import data
import json
import numpy as np
import math

IMAGE_DIM = 112
MARGIN = 3

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

datasets = data('datasets')['Item']
count = 0
for item in datasets:
    count += 1
    if count > 50:
        break
    dataset = data(item)
    columns = []
    for column_name in dataset:
        if dataset[column_name].dtype != np.object:
            columns.append(dataset[column_name].tolist())
    if len(columns) < 2:
        continue
    for i in range(len(columns)):
        for j in range(1, len(columns)):
            scatter = gen_scatter(columns[i], columns[j])
            if scatter != None:
                scatters.append(scatter)
            d = gen_image(columns[i], columns[j])
            if d != None:
                image, scatter = d
                images.append(image)
                scatters.append(scatter)
            

with open("data/scatters_" + str(IMAGE_DIM) + ".json", "w") as f:
    json.dump(images, f)

with open("data/scatters_points_" + str(IMAGE_DIM) + ".json", "w") as f:
    json.dump(scatters, f, separators=(',', ':'), indent=4)
