# Imports
import torch
cuda = torch.cuda.is_available()
import numpy as np
import sys
sys.path.append("m1m2/semi-supervised")


num_epoch = 1010

batch_size = 64

img_dim = 112
# 分类数
y_dim = 6
# 潜空间数
z_dim = 32
# 隐藏层配置
h_dim = [2048, 1024, 512, 256, 128]
# 输入空间数
x_dim = img_dim ** 2


from models import VariationalAutoencoder
from layers import GaussianSample
model = VariationalAutoencoder([x_dim, z_dim, h_dim])
model




from torch.autograd import Variable

gaussian = GaussianSample(z_dim, 1)
z, mu, log_var = gaussian(Variable(torch.ones(1, z_dim)))

print(f"sample {float(z.data):.2f} drawn from N({float(mu.data):.2f}, {float(log_var.exp().data):.2f})")

print(model._kld.__doc__)




from datautils import get_mnist, get_scatters

_, train, validation = get_mnist(location="./", batch_size=64)

_, train, validation = get_scatters("data/images_generated.json", "data/labels_generated.json", 300, batch_size, 100, y_dim, cuda)

# We use this custom BCE function until PyTorch implements reduce=False
def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))




for epoch in range(50):
    model.train()
    total_loss = 0
    for (u, _) in train:
        u = Variable(u)

        if cuda: u = u.cuda(device=0)

        reconstruction = model(u)
        
        likelihood = -binary_cross_entropy(reconstruction, u)
        elbo = likelihood - model.kl_divergence
        
        L = -torch.mean(elbo)

        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += L.data[0]

    m = len(train)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}\tL: {total_loss/m:.2f}")



# model.eval()
# x_mu = model.sample(Variable(torch.randn(16, 32)))




# f, axarr = plt.subplots(1, 16, figsize=(18, 12))

# samples = x_mu.data.view(-1, 28, 28).numpy()

# for i, ax in enumerate(axarr.flat):
#     ax.imshow(samples[i])
#     ax.axis("off")