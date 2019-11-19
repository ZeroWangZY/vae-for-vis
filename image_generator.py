from pydataset import data
import json
import numpy as np

def gen_image(arr1, arr2):
    IMAGE_DIM = 200
    MARGIN = 10
    # init image
    result = [[0 for col in range(IMAGE_DIM)] for row in range(IMAGE_DIM)]

    min1 = min(arr1)
    max1 = max(arr1)
    min2 = min(arr2)
    max2 = max(arr2)
    minP = MARGIN
    maxP = IMAGE_DIM - MARGIN
    for i in range(len(arr1)):
        try:
            x = round(minP + ((maxP - minP) * (arr1[i] - min1)) / (max1 - min1))
            y = round(minP + ((maxP - minP) * (arr2[i] - min2)) / (max2 - min2))
            for j in range(x - 2, x + 3):
                for k in range(y - 2, y + 3):
                    result[j][k] += 1
        except ValueError:
            print('ValueError')
        except ZeroDivisionError:
            print('ZeroDivisionError')
        result = np.array(result)
    return (result / np.max(result)).tolist()


images = []
datasets = data('datasets')['Item']
count = 0
for item in datasets:
    count += 1
    if count > 20:
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
            images.append(gen_image(columns[i], columns[j]))

with open("data/scatters.json", "w") as f:
    json.dump(images, f)


