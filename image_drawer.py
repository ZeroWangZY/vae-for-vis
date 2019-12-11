import numpy as np
import imageio, os
import json

img_dim = 112

use_real = True

file_prefix = "real_" if use_real else ""

def gen_scatters_image(scatters_data, path='scatters.png'):
    n = 20
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            length = len(scatters_data)
            digit = 0 if length == 0 else scatters_data[np.random.choice(length)]
            figure[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) *
                   img_dim] = digit
    imageio.imwrite(path, (figure * 255))


scatters_data = []
with open("data/" + file_prefix + "scatters_" + str(img_dim) + ".json", 'r') as load_f:
    scatters_data = json.load(load_f)
print(len(scatters_data))
print(np.max(scatters_data), np.min(scatters_data))
gen_scatters_image(scatters_data,
                   path='image/origin_scatters_sample_' + str(img_dim) + '.png')
