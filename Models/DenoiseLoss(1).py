import torch
import torch.nn as nn

class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert(hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.pow(torch.abs(x - y), self.n) / self.n
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]

        loss = loss.mean()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss