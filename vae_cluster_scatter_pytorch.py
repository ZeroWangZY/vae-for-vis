from __future__ import print_function

import argparse
import json

import imageio
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='VAE cluster')
parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=1010,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
latent_dim = 20
num_classes = 9
img_dim = 112
intermediate_dim = 256
# 无标签与有标签的比例
label_ratio = 1
unlabel_ratio = 10
batch_size = args.batch_size
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

scatters_data = []
scatters_label = []

# with open("data/scatters_" + str(img_dim) + ".json", 'r') as load_f:
#     scatters_data = json.load(load_f)

# with open("data/scatters_labels_" + str(img_dim) + ".json", 'r') as load_f:
#     scatters_label_info = json.load(load_f)

with open("data/images_generated.json", 'r') as load_f:
    scatters_data = json.load(load_f)

with open("data/labels_generated.json", 'r') as load_f:
    scatters_label_info = json.load(load_f)

scatters_label = []
for _, scatter_info in enumerate(scatters_label_info):
    scatters_label.append(scatter_info["label"]) 

x_train = torch.Tensor(scatters_data)
x_train_numpy = np.array(scatters_data)
x_train = torch.unsqueeze(x_train, 1)

# 要转成long，要不求交叉熵报类型错误
y_train = torch.Tensor(scatters_label).long()
y_train_numpy = np.array(scatters_label)

# 先转换成 torch 能识别的 Dataset
torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)

train_loader = torch.utils.data.DataLoader(torch_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)
x_train = x_train.cuda()
x_test = x_train[:128]


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        # TODO: padding: same?
        self.mu_layer = nn.Linear(746496, latent_dim)
        self.logvar_layer = nn.Linear(746496, latent_dim)

        self.linear1 = nn.Linear(latent_dim, 746496)
        self.convT1 = nn.ConvTranspose2d(64, 32, 3, stride=1)
        self.convT2 = nn.ConvTranspose2d(32, 1, 3, stride=1)

        self.linear1_y = nn.Linear(latent_dim, intermediate_dim)
        self.linear2_y = nn.Linear(intermediate_dim, num_classes)

        self.gaussian_mean = nn.Parameter(
            torch.zeros((num_classes, latent_dim)))

    def encode(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.conv2(h1))  # h2.shape [128, 64, 108, 108]
        h2 = h2.reshape(h2.shape[0], -1)
        return self.mu_layer(h2), self.logvar_layer(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.linear1(z))
        h3 = h3.reshape(-1, 64, 108, 108)
        h4 = F.leaky_relu(self.convT1(h3))
        h5 = F.leaky_relu(self.convT2(h4))
        return torch.sigmoid(h5)

    def classifier(self, z):
        h1 = F.relu(self.linear1_y(z))
        h2 = self.linear2_y(h1)
        y = F.softmax(h2,dim=1)
        return y

    def classify(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.classifier(z)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.classifier(z)
        z_prior_mean = torch.unsqueeze(z, 1) - torch.unsqueeze(
            self.gaussian_mean, 0)
        return self.decode(z), mu, logvar, y, z_prior_mean


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# writer.close()

def calcLabelLoss(y, label, unlabel_ratio, label_ratio):
    num_of_data = y.size()[0]
    size_of_tuple = unlabel_ratio + label_ratio
    num_of_tuple = num_of_data // size_of_tuple

    label_loss = 0
    loss_func = nn.CrossEntropyLoss()
    for i in range(num_of_tuple):
        start_index = i * size_of_tuple
        end_index = start_index + label_ratio

        sub_y = y[start_index: end_index]
        sub_label = label[start_index: end_index]

        label_loss = label_loss + loss_func(sub_y, sub_label)

    return label_loss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_recon, x, mu, z_log_var, y, z_prior_mean, label):
    z_log_var = torch.unsqueeze(z_log_var, 1)
    lamb = 2.5  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
    xent_loss = 0.5 * torch.mean(
        (x.view(-1, img_dim**2) - x_recon.view(-1, img_dim**2))**2, 0)
    kl_loss = -0.5 * (z_log_var - torch.mul(z_prior_mean, z_prior_mean))
    kl_loss = torch.mean(torch.matmul(torch.unsqueeze(y, 1), kl_loss), 0)
    cat_loss = torch.mean(y * torch.log(y + 1e-07), 0)
    label_loss = calcLabelLoss(y, label, unlabel_ratio, label_ratio)
    
    vae_loss = lamb * torch.sum(xent_loss) + torch.sum(kl_loss) + torch.sum(
        cat_loss) + label_loss
    return vae_loss, torch.sum(xent_loss), torch.sum(kl_loss), torch.sum(
        cat_loss), label_loss


def train(epoch, writer):
    model.train()
    train_loss = 0
    train_xent_loss = 0
    train_kl_loss = 0
    train_cat_loss = 0
    train_label_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)  # data.shape = [batch_size, 1, 28, 28]
        label = label.to(device)
        if epoch == 1:
            writer.add_graph(model, data)
        optimizer.zero_grad()
        recon_batch, mu, logvar, y, z_prior_mean = model(data)
        loss, xent_loss, kl_loss, cat_loss, label_loss = loss_function(
            recon_batch, data, mu, logvar, y, z_prior_mean, label)
        loss.backward()
        train_loss += loss.item()
        train_xent_loss += xent_loss.item()
        train_kl_loss += kl_loss.item()
        train_cat_loss += cat_loss.item()
        train_label_loss += label_loss.item()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / len(data)))
    writer.add_scalar('loss/loss', train_loss / len(train_loader.dataset),
                      epoch)
    writer.add_scalar('loss/kl_loss',
                      train_kl_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('loss/xent_loss',
                      train_xent_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('loss/cat_loss',
                      train_cat_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('loss/label_loss',
                      train_label_loss / len(train_loader.dataset), epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def cluster_sample(y_train_pred, category=0, n=5):
    y_train_pred = np.array(y_train_pred)
    figure = np.zeros((img_dim * n, img_dim * n))
    idxs = np.where(y_train_pred == category)[0]
    if len(idxs) == 0:
        return figure
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index >= len(idxs):
                break
            digit = x_train_numpy[idxs[index]]
            digit = digit.reshape((img_dim, img_dim))
            figure[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) *
                   img_dim] = digit
    output = figure * 255
    return output


def draw_cluster_sample_to_png(y_train_pred, path, sub_width=5):
    n = 0
    while n * n < num_classes:
        n += 1

    image = np.zeros((img_dim * n * sub_width, img_dim * n * sub_width))

    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index >= num_classes:
                break
            sub_image = cluster_sample(y_train_pred,
                                       category=index,
                                       n=sub_width)
            image[img_dim * i * sub_width:img_dim * (i + 1) *
                  sub_width, img_dim * j * sub_width:img_dim * (j + 1) *
                  sub_width] = sub_image
    imageio.imwrite(path, image)


def test_and_save(epoch):
    ys = None
    zs = None
    for i in range(1000):
        batch = args.epochs
        if batch_size * (i + 1) >= x_train_numpy.shape[0]:
            mu, logvar = model.encode(x_train[batch_size *
                                              i:x_train_numpy.shape[0]])
            z = model.reparameterize(mu, logvar)
            y = torch.argmax(model.classifier(z), dim=1)
            if i == 0:
                ys = y
                zs = z
            else:
                ys = torch.cat((ys, y), 0)
                zs = torch.cat((zs, z), 0)
            break
        mu, logvar = model.encode(x_train[batch_size * i:batch_size * (i + 1)])
        z = model.reparameterize(mu, logvar)
        y = torch.argmax(model.classifier(z), dim=1)
        if i == 0:
            ys = y
            zs = z
        else:
            ys = torch.cat((ys, y), 0)
            zs = torch.cat((zs, z), 0)
    # draw_cluster_sample_to_png(ys.cpu().tolist(),
    #                            'results/' + str(epoch) + '.png',
    #                            sub_width=5)
    writer.add_embedding(zs,
                         metadata=ys.cpu().tolist(),
                         label_img=x_train,
                         global_step=epoch)


if __name__ == "__main__":
    writer = SummaryWriter(log_dir='runs/semisupervised-1')
    for epoch in range(1, args.epochs + 1):
        train(epoch, writer)
        if epoch % 10 == 0:
            with torch.no_grad():
                test_and_save(epoch)
