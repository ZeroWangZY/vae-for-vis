import json
import numpy as np
import math
import imageio


path_to_read = "data/scatters_generated.json"
path_to_write = "data/images_generated.json"
path_to_write_imgs = "data/png2/"
IMAGE_DIM = 112
MARGIN = 3


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def isItemValid(x, y):
    return (not math.isnan(x)) and (not math.isnan(y))


def gen_scatters_image(scatters_data, path):
    figure = np.zeros((IMAGE_DIM, IMAGE_DIM))
    figure[0: IMAGE_DIM, 0: IMAGE_DIM] = scatters_data
    imageio.imwrite(path, figure * 255)


def gen_image(arr1, arr2):

    # init image
    result = [[0 for col in range(IMAGE_DIM)] for row in range(IMAGE_DIM)]

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
            for j in range(x - 2, x + 3):
                for k in range(y - 2, y + 3):
                    result[j][k] += 1
        except (ValueError, ZeroDivisionError):
            pass
        result = np.array(result)
    if np.max(result) <= 0:
        return None
    result = result / np.max(result)
    return result.tolist()



def dict2arr(dict_arr):
    keys = list(dict_arr[0].keys())
    length_of_keys = len(keys)
    length_of_dict_arr = len(dict_arr)
    ret = [[] for i in range(length_of_keys)]
    for i in range(length_of_dict_arr):
        item = dict_arr[i]
        for j in range(length_of_keys):
            ret[j].append(item[keys[j]])
    return ret



def main():
    with open(path_to_read, 'r') as load_f:
        dataset = json.load(load_f)

    images = []
    class_name = list(dataset.keys())
    num_of_class = len(class_name)
    for i in range(num_of_class):
        scatters = dataset[class_name[i]]
        num_of_instance = len(scatters)
        for j in range(num_of_instance):
            scatter = scatters[j]
            [x_arr, y_arr, *_] = dict2arr(scatter)
            image = gen_image(x_arr, y_arr)
            gen_scatters_image(image, path_to_write_imgs + class_name[i] + "_" + str(j) + ".png")
            images.append(image)

    with open(path_to_write, "w") as f:
        json.dump(images, f)

if __name__ == "__main__":
    main()