"""Two 256-wide ReLU hidden layers: scalar Q(s,a)."""
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def soft_update(target, online, tau):
    with torch.no_grad():
        for dest, src in zip(target.parameters(), online.parameters()):
            dest.mul_(1 - tau).add_(src, alpha=tau)
