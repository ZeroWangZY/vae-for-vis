import json
import random
import math

path_to_write = "data/scatters_generated.json"

def gen_random_in_domain(domain):
    t = random.random()
    return t * domain[1] + (1 - t) * domain[0]


def is_in_domain(point, domain):
    length = len(point)
    for i in range(length):
        coord = point[i]
        dimension = domain[i]
        if (coord < dimension[0] or coord > dimension[1]):
            return False
    return True

def gen_circular_cluster_sample(radius, center):
    r = radius * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    x = r * math.cos(theta) + center[0]
    y = r * math.sin(theta) + center[1]
    return x, y


def gen_circular_cluster(domain, center, radius, num_per_plot):
    ret_cluster = []
    for i in range(0, num_per_plot):
        x, y = gen_circular_cluster_sample(radius, center)
        while (not is_in_domain([x, y], domain)):
            x, y = gen_circular_cluster_sample(radius, center)
        ret_cluster.append({
          "x": x,
          "y": y
        })
    return ret_cluster


def gen_ellipse_cluster_sample(A, B, center):
    a = random.uniform(0, A)
    b = random.uniform(0, B)
    theta = 2 * math.pi * random.random() - math.pi
    # 将极坐标转换成笛卡尔坐标，并做相关平移
    x = a * math.cos(theta)
    y = b * math.sin(theta)
    angle = math.pi / 4
    x1 = x * math.cos(angle) - y * math.sin(angle) + center[0]
    y1 = x * math.sin(angle) + y * math.cos(angle) + center[1]
    return x1, y1


def gen_ellipse_cluster(domain, center, long_axis_len, short_axis_len, num_per_plot):
    ret_cluster = []
    A = long_axis_len / 2
    B = short_axis_len / 2
    for i in range(num_per_plot):
        x1, y1 = gen_ellipse_cluster_sample(A, B, center)
        while (not is_in_domain([x1, y1], domain)):
            x1, y1 = gen_ellipse_cluster_sample(A, B, center)  
        ret_cluster.append({
          "x": x1,
          "y": y1
        })
    return ret_cluster


def gen_gaussian_cluster_sample(mu, sigma, domain):
    [x_domain, y_domain] = domain
    x = gen_random_in_domain(x_domain)
    y = math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return x, y


def gen_gaussian_cluster(domain, mu, sigma, num_per_plot):
    ret_cluster = []
    for i in range(num_per_plot):
        x, y = gen_gaussian_cluster_sample(mu, sigma, domain)
        while (not is_in_domain([x, y], domain)):
            x, y = gen_gaussian_cluster_sample(mu, sigma, domain)
        ret_cluster.append({
          "x": x,
          "y": y
        })
    return ret_cluster


def gen_sin_cluster_sample(amplitude, omega, phi, center, domain):
    [x_domain, y_domain] = domain
    x1 = gen_random_in_domain(x_domain)
    y1 = amplitude * math.sin(omega * (x1 + phi))
    x = x1 + center[0]
    y = y1 + center[1]
    return x, y


def gen_sin_cluster(domain, center, amplitude, omega, phi, num_per_plot):
    ret_cluster = []
    for i in range(num_per_plot):
        x, y = gen_sin_cluster_sample(amplitude, omega, phi, center, domain)
        while (not is_in_domain([x, y], domain)):
            x, y = gen_sin_cluster_sample(amplitude, omega, phi, center, domain)
        ret_cluster.append({
          "x": x,
          "y": y
        })
    return ret_cluster


def gen_cluster1(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    max_x_radius = (x_domain[1] - x_domain[0]) / 2
    max_y_radius = (y_domain[1] - y_domain[0]) / 2
    max_radius = min(max_x_radius, max_y_radius)
    for i in range(num_per_class):
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        radius = random.random() * max_radius
        x_center = gen_random_in_domain(x_domain)
        y_center = gen_random_in_domain(y_domain)

        ret_data.append(gen_circular_cluster([x_domain, y_domain], [x_center, y_center], radius, num_per_plot))
    return ret_data



def gen_cluster2(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    max_x_radius = (x_domain[1] - x_domain[0]) / 4
    max_y_radius = (y_domain[1] - y_domain[0]) / 4
    max_radius = min(max_x_radius, max_y_radius)
    for i in range(num_per_class):
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        radius1 = random.random() * max_radius
        x1_domain_of_center = [x_domain[0] + radius1, x_domain[1] - radius1]
        y1_domain_of_center = [y_domain[0] + radius1, y_domain[1] - radius1]
        x1_center = gen_random_in_domain(x1_domain_of_center)
        y1_center = gen_random_in_domain(y1_domain_of_center)
        cluster1 = gen_circular_cluster([x_domain, y_domain], [x1_center, y1_center], radius1, num_per_plot // 2)

        radius2 = random.random() * max_radius
        x2_domain_of_center = [x_domain[0] + radius2, x_domain[1] - radius2]
        y2_domain_of_center = [y_domain[0] + radius2, y_domain[1] - radius2]
        x2_center = gen_random_in_domain(x2_domain_of_center)
        y2_center = gen_random_in_domain(y2_domain_of_center)
        cluster2 = gen_circular_cluster([x_domain, y_domain], [x2_center, y2_center], radius2, num_per_plot // 2)

        ret_data.append(cluster1 + cluster2)
    return ret_data



def gen_cluster3(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    max_x_radius = (x_domain[1] - x_domain[0]) / 6
    max_y_radius = (y_domain[1] - y_domain[0]) / 6
    max_radius = min(max_x_radius, max_y_radius)
    for i in range(num_per_class):
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        radius1 = random.random() * max_radius
        x1_domain_of_center = [x_domain[0] + radius1, x_domain[1] - radius1]
        y1_domain_of_center = [y_domain[0] + radius1, y_domain[1] - radius1]
        x1_center = gen_random_in_domain(x1_domain_of_center)
        y1_center = gen_random_in_domain(y1_domain_of_center)
        cluster1 = gen_circular_cluster([x_domain, y_domain], [x1_center, y1_center], radius1, num_per_plot // 3)

        radius2 = random.random() * max_radius
        x2_domain_of_center = [x_domain[0] + radius2, x_domain[1] - radius2]
        y2_domain_of_center = [y_domain[0] + radius2, y_domain[1] - radius2]
        x2_center = gen_random_in_domain(x2_domain_of_center)
        y2_center = gen_random_in_domain(y2_domain_of_center)
        cluster2 = gen_circular_cluster([x_domain, y_domain], [x2_center, y2_center], radius2, num_per_plot // 3)

        radius3 = random.random() * max_radius
        x3_domain_of_center = [x_domain[0] + radius3, x_domain[1] - radius3]
        y3_domain_of_center = [y_domain[0] + radius3, y_domain[1] - radius3]
        x3_center = gen_random_in_domain(x3_domain_of_center)
        y3_center = gen_random_in_domain(y3_domain_of_center)
        cluster3 = gen_circular_cluster([x_domain, y_domain], [x3_center, y3_center], radius3, num_per_plot // 3)

        ret_data.append(cluster1 + cluster2 + cluster3)
    return ret_data



def gen_corr(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    center = [(x_domain[0] + x_domain[1]) / 2, (y_domain[0] + y_domain[1]) / 2]
    x_len = x_domain[1] - x_domain[0]
    y_len = y_domain[1] - y_domain[0]
    max_diagonal_len = math.sqrt(x_len ** 2 + y_len ** 2)
    max_back_diagonal_len = math.sqrt(x_len ** 2 + y_len ** 2)
    split_ratio = 0.5
    for i in range(num_per_class):
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        diagonal_len = gen_random_in_domain([split_ratio * max_diagonal_len, max_diagonal_len])
        back_diagonal_len = gen_random_in_domain([0 * max_back_diagonal_len, split_ratio * max_back_diagonal_len])
        ret_data.append(gen_ellipse_cluster([x_domain, y_domain], center, diagonal_len, back_diagonal_len, num_per_plot))
    return ret_data



def gen_normal(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    sigma_domain = [0, 1]
    for i in range(num_per_class):   
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        mu = gen_random_in_domain(x_domain)
        sigma = gen_random_in_domain(sigma_domain)
        ret_data.append(gen_gaussian_cluster([x_domain, y_domain], mu, sigma, num_per_plot))
    return ret_data



def gen_sin(x_domain, y_domain, num_per_class, max_num_plot):
    ret_data = []
    amplitude_range = [0, (x_domain[1] - x_domain[0]) / 2]
    omega_range = [1 * math.pi, 20 * math.pi]
    phi_range = [0, 2 * math.pi]
    center = [(x_domain[0] + x_domain[1]) / 2, (y_domain[0] + y_domain[1]) / 2]
    for i in range(num_per_class):
        num_per_plot = round(gen_random_in_domain(max_num_plot))
        amplitude = gen_random_in_domain(amplitude_range)
        omega = gen_random_in_domain(omega_range)
        phi = gen_random_in_domain(phi_range)
        ret_data.append(gen_sin_cluster([x_domain, y_domain], center, amplitude, omega, phi, num_per_plot))
    return ret_data



def main():
    x_domain = [0, 1]
    y_domain = x_domain
    num_per_class = 500
    max_num_plot = [100, 1000]
    class_categories = ["cluster1", "cluster2", "cluster3", "corr", "normal", "sin"]
    func_deliver = {
      "cluster1": gen_cluster1,
      "cluster2": gen_cluster2,
      "cluster3": gen_cluster3,
      "corr": gen_corr,
      "normal": gen_normal,
      "sin": gen_sin
    }

    dataset = {}
    for (idx, class_name) in enumerate(class_categories):
        print("generating class as {} : {} / {}".format(class_name, idx, len(class_categories)))
        generator = func_deliver[class_name]
        dataset[class_name] = generator(x_domain, y_domain, num_per_class, max_num_plot)

    with open(path_to_write, "w") as f:
        json.dump(dataset, f)
        # json.dump(dataset, f, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    main()