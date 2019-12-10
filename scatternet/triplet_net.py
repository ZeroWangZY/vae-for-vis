import torch.nn as nn
import torch.nn.functional as F


def normalize(embedding):
    return F.normalize(embedding, p = 2, dim = 1)


class TripletNet(nn.Module):
    def __init__(self, embedding_net, vector_dist_func = None):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.vector_dist_func = vector_dist_func

    def forward(self, anchor_data, pos_data, neg_data):
        anchor_embedding = self.embedding_net(anchor_data)
        pos_embedding = self.embedding_net(pos_data)
        neg_embedding = self.embedding_net(neg_data)

        # 对embedding做L2归一化
        anchor_embedding = normalize(anchor_embedding)
        pos_embedding = normalize(pos_embedding)
        neg_embedding = normalize(neg_embedding)

        # 算2个距离，这里用欧氏距离
        if self.vector_dist_func is None:
            dist_pos = F.pairwise_distance(anchor_embedding, pos_embedding, 2).pow(2)
            dist_neg = F.pairwise_distance(anchor_embedding, neg_embedding, 2).pow(2)
        else:
            dist_pos = self.vector_dist_func(anchor_embedding, pos_embedding)
            dist_neg = self.vector_dist_func(anchor_embedding, neg_embedding)

        return dist_pos, dist_neg, anchor_embedding, pos_embedding, neg_embedding