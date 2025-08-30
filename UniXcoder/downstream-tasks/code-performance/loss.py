import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, preds, targets):
        P_y = F.softmax(targets, dim=0)
        P_s = F.softmax(preds, dim=0)
        loss = -(P_y * torch.log(P_s + 1e-12)).sum()
        return loss