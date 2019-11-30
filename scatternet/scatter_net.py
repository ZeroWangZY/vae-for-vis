import torch
import torch.nn as nn
from triplet_net import TripletNet
from cnn_net import CnnNet
from torchvision import transforms
from torch.autograd import Variable
from triplet_mnist_loader import MNIST_t
from tools import VisdomLinePlotter, AverageMeter


def main():
    # 一堆常量
    epochs = 1
    batch_size = 5000
    learning_rate = 0.01
    triplet_margin = 0.5
    log_interval = 5 # 多久打印1次
    model_path = "./model/scatter_net.pkl"

    hyper_param = {
      "epochs": epochs,
      "batch_size": batch_size,
      "learning_rate": learning_rate,
      "triplet_margin": triplet_margin,
      "log_interval": log_interval,
      "model_path": model_path
    }


    scatter_net = TripletNet(CnnNet())
    print("archtitecture of ScatterNet:\n", scatter_net)

    train_loader = torch.utils.data.DataLoader(
        MNIST_t('./data',
            train = True, 
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size = batch_size,
        shuffle = True,
        num_workers = 3
    )

    test_loader = torch.utils.data.DataLoader(
        MNIST_t('./data',
            train = False, 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size = batch_size,
        shuffle = True
    )

    optimizer = torch.optim.Adam(scatter_net.parameters(), lr = learning_rate)

    for epoch in range(1, epochs + 1):
        train(epoch, train_loader, scatter_net, optimizer, hyper_param)
        acc = test(epoch, test_loader, scatter_net, hyper_param)

    save(scatter_net, model_path)



def calcTripletLoss(dist_pos, dist_neg, triplet_margin):
    criterion = torch.nn.MarginRankingLoss(margin = triplet_margin)

    # all -1s means: If y = 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y = -1.
    # https://pytorch.org/docs/stable/nn.html?highlight=marginrankingloss#torch.nn.MarginRankingLoss
    labelHelper = torch.FloatTensor(dist_pos.size()).fill_(-1)
    labelHelper = Variable(labelHelper)

    triplet_loss = criterion(dist_pos, dist_neg, labelHelper)
    return triplet_loss



def calcAccuracy(dist_pos, dist_neg, triplet_margin):
    predictions = torch.gt(dist_neg, dist_pos + triplet_margin).double()
    return (predictions > 0).sum() / dist_pos.size()[0]




def train(epoch, train_loader, scatter_net, optimizer, hyper_param):
    # 设置模型模式，就是设个标志位，其它地方有用，例如dropout
    scatter_net.train()
    triplet_margin = hyper_param["triplet_margin"]
    log_interval = hyper_param["log_interval"]

    # 开始训练
    for batch_idx, (anchor_data, pos_data, neg_data) in enumerate(train_loader):
        anchor_data, pos_data, neg_data = Variable(anchor_data), Variable(pos_data), Variable(neg_data)

        # 正向传播
        dist_pos, dist_neg, anchor_embedding, pos_embedding, neg_embedding = scatter_net(anchor_data, pos_data, neg_data)

        triplet_loss = calcTripletLoss(dist_pos, dist_neg, triplet_margin)
        # 可能还有其它的loss项，这里先不加
        loss = triplet_loss
        loss_for_log = loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f}\t'
                .format(
                    epoch, batch_idx * len(anchor_data), len(train_loader.dataset),
                    loss_for_log
                )
            )


def test(epoch, test_loader, scatter_net, hyper_param):
    losses = AverageMeter()
    accs = AverageMeter()

    triplet_margin = hyper_param["triplet_margin"]

    # 关闭训练模式
    scatter_net.eval()
    for batch_idx, (anchor_data, pos_data, neg_data) in enumerate(test_loader):
        anchor_data, pos_data, neg_data = Variable(anchor_data), Variable(pos_data), Variable(neg_data)

        # 正向传播
        dist_pos, dist_neg, anchor_embedding, pos_embedding, neg_embedding = scatter_net(anchor_data, pos_data, neg_data)

        triplet_loss = calcTripletLoss(dist_pos, dist_neg, triplet_margin)
        # 可能还有其它的loss项，这里先不加
        loss = triplet_loss
        loss_for_log = loss.item()

        acc = calcAccuracy(dist_pos, dist_neg, triplet_margin)
        accs.update(acc, anchor_data.size(0))
        losses.update(loss_for_log, anchor_data.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg



def save(scatter_net, model_path):
    torch.save(scatter_net, model_path)  # 保存整个网络


def restore_net():
    scatter_net = torch.load(model_path)
    return scatter_net

if __name__ == "__main__":
    main()