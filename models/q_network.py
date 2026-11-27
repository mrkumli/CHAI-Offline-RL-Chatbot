"""Two Q_HIDDEN-wide (256) ReLU hidden layers: scalar Q(s,a)."""
import torch
from torch import nn
from config import Q_HIDDEN


class QNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, Q_HIDDEN), nn.ReLU(),
                                 nn.Linear(Q_HIDDEN, Q_HIDDEN), nn.ReLU(), nn.Linear(Q_HIDDEN, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def soft_update(target, online, tau):
    with torch.no_grad():
        for dest, src in zip(target.parameters(), online.parameters()):
            dest.mul_(1 - tau).add_(src, alpha=tau)
