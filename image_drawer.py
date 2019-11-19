import numpy as np
import imageio, os
import json


def gen_scatters_image(scatters_data, path='scatters.png'):
    img_dim = len(scatters_data[0])
    n = 10
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            digit = scatters_data[np.random.choice(len(scatters_data))]
            figure[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) *
                   img_dim] = digit
    imageio.imwrite(path, (figure * 255))


scatters_data = []
with open("data/scatters_64.json", 'r') as load_f:
    scatters_data = json.load(load_f)
print(len(scatters_data))
print(np.max(scatters_data), np.min(scatters_data))
gen_scatters_image(scatters_data, path='image/origin_scatters_sample_64.png')
