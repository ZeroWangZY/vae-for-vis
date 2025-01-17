# Imports
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/semisupervised-ladder')
cuda = torch.cuda.is_available()
import numpy as np


import sys
sys.path.append("m1m2/semi-supervised")



from models import LadderDeepGenerativeModel

num_epoch = 100

batch_size = 64

img_dim = 112
# 分类数
y_dim = 6
# 潜空间数, 是一个数组，长度为输入层层数 + 隐藏层层数
z_dim = [32, 16, 8]
# 隐藏层配置
h_dim = [1024, 512, 128]
# 输入空间数
x_dim = img_dim ** 2

num_valid = 1000

labels_per_class = 100

model = LadderDeepGenerativeModel([x_dim, y_dim, z_dim, h_dim])



from datautils import get_mnist, get_scatters

# Only use 10 labelled examples per class
# The rest of the data is unlabelled.
# labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)
labelled, unlabelled, validation = get_scatters("data/images_generated.json", "data/labels_generated.json", num_valid, batch_size, labels_per_class, y_dim, cuda)
alpha = 0.1 * (len(unlabelled) + len(labelled)) / len(labelled)

def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))





from itertools import cycle
from inference import SVI, DeterministicWarmup

# We will need to use warm-up in order to achieve good performance.
# Over 200 calls to SVI we change the autoencoder from
# deterministic to stochastic.
beta = DeterministicWarmup(n=200)

if cuda: model = model.cuda()
elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta)





from torch.autograd import Variable

for epoch in range(1, num_epoch + 1):
    model.train()
    total_loss, accuracy = (0, 0)
    for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
        # Wrap in variables
        x, y, u = Variable(x), Variable(y), Variable(u)

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)

        L = -elbo(x, y)
        U = -elbo(u)

        # Add auxiliary classification loss q(y|x)
        logits = model.classify(x)
        
        # Regular cross entropy
        classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss + U

        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

    if epoch % 1 == 0:
        model.eval()
        m = len(unlabelled)
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        writer.add_scalar("loss/train_loss", total_loss / m, epoch)
        writer.add_scalar("accu/train_accu", accuracy / m, epoch)

        total_loss, accuracy = (0, 0)

        mat = None
        metadata = None
        label_img = None

        cnt = 0
        for x, y in validation:
            cur_batch_size = x.size()[0]
            x, y = Variable(x), Variable(y)

            if cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            U = -elbo(x)

            # ladder-vae的隐变量要看下
            z = model.sample_z()

            logits = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L + alpha * classication_loss + U

            total_loss += J_alpha.item()

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

            img = x.reshape(cur_batch_size, -1, img_dim, img_dim)
            if cnt == 0:
                metadata = pred_idx
                label_img = img
                mat = z
            else:
                metadata = torch.cat((metadata, pred_idx), 0)
                label_img = torch.cat((label_img, img), 0)
                mat = torch.cat((mat, z), 0)
            cnt += 1

        m = len(validation)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
        writer.add_scalar("loss/test_loss", total_loss / m, epoch)
        writer.add_scalar("accu/test_accu", accuracy / m, epoch)

    if epoch % 10 == 0:
        writer.add_embedding(mat=mat,
            metadata=metadata.cpu().tolist(),
            label_img=label_img,
            global_step=epoch)