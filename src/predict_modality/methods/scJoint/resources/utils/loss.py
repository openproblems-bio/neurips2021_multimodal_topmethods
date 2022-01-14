""" define custum loss function in this file """
import torch
import torch.nn as nn


def cosine_sim(arr_1, arr_2):
    """ return consine similarity of 2 arrays """
    arr_1 = arr_1 / torch.norm(arr_1, dim=1, keepdim=True)
    arr_2 = arr_2 / torch.norm(arr_2, dim=1, keepdim=True)
    sim = torch.matmul(arr_1, torch.transpose(arr_2, 0, 1))

    return sim


class CosineLoss(nn.Module):
    """ custum loss for mean cosine similarity """
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, emb1, emb2, emb1_resid, emb2_resid):
        """ define cosine loss """
        emb1, emb2 = emb1.float(), emb2.float()
        cosine_loss = torch.mean(
            torch.abs(cosine_sim(emb1, emb1_resid) + cosine_sim(emb2, emb2_resid))
        )
        return cosine_loss


class L1regularization(nn.Module):
    """ l1 regularization loss for model """
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        """ define l1 reg loss """
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss
