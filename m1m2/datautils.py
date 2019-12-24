import json
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import onehot
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from operator import itemgetter, __or__
from torchvision.datasets import MNIST
from functools import reduce
import sys


def get_sampler(labels, n_labels, n=None):
    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler




def get_mnist(location="./", batch_size=64, labels_per_class=100, n_labels=10, cuda=False):
    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    # print("111", type(mnist_train.train_labels.numpy()))
    # labels = torch.tensor([1,2,3,0])
    # onehot_converter = onehot(4)
    # labels = labels.map_(transforms.Lambda(onehot_converter))
    # print(labels)
    # sys.exit(0)
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), n_labels, labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy(), n_labels))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy(), n_labels))

    return labelled, unlabelled, validation



class ScattersDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, image_tensors, label_tensors, transform=None, target_transform=None):
        self.image_tensors = image_tensors
        self.label_tensors = label_tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.image_tensors[index]

        if self.transform:
            x = self.transform(x)

        y = self.label_tensors[index]

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.image_tensors.size(0)




def split_scatters(images, labels, num_valid, random_seed):
    num_dataset = len(images)
    num_train = num_dataset - num_valid
    indices = list(range(num_dataset))
    split = num_valid

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    # print(train_idx, valid_idx)
    train_images = list(itemgetter(*train_idx)(images))
    train_labels = list(itemgetter(*train_idx)(labels))

    train_images = torch.tensor(train_images)
    train_images = torch.flatten(train_images, start_dim=1)
    train_labels = torch.tensor(train_labels)

    valid_images = list(itemgetter(*valid_idx)(images))
    valid_labels = list(itemgetter(*valid_idx)(labels))

    valid_images = torch.tensor(valid_images)
    valid_images = torch.flatten(valid_images, start_dim=1)
    valid_labels = torch.tensor(valid_labels)

    return [train_images, train_labels], [valid_images, valid_labels]




def get_scatters(image_location="data/images_generated.json", label_location="data/labels_generated.json", num_valid=300, batch_size=64, labels_per_class=100, n_labels=6, cuda=False):
    torch.manual_seed(1)
    random_seed = 1    # reproducible

    with open(image_location, 'r') as load_f:
        images = json.load(load_f)
    with open(label_location, 'r') as load_f:
        labels = json.load(load_f)
    labels = [item["label"] for item in labels]

    scatters_train, scatters_valid = split_scatters(images, labels, num_valid, random_seed)

    onehot_converter = onehot(n_labels)

    data_train = ScattersDataset(image_tensors=scatters_train[0], label_tensors=scatters_train[1], target_transform=onehot_converter)

    data_valid = ScattersDataset(image_tensors=scatters_valid[0], label_tensors=scatters_valid[1], target_transform=onehot_converter)

    labelled = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(data_train.label_tensors.numpy(), n_labels, labels_per_class))
    unlabelled = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(data_train.label_tensors.numpy(), n_labels))
    validation = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(data_valid.label_tensors.numpy(), n_labels))

    return labelled, unlabelled, validation