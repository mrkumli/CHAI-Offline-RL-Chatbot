"""CHAI-BRAC: conditional Gaussian price policy with KL to behavior Gaussian.

Paper Sec. 4.1: max E[Q(s,a')] - KL(pi_phi(price|s) || pi_B(price|s)).
Price prior mean & standard deviation are linear in previous offer.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


class PricePrior(nn.Module):
    def __init__(self, mean_coeff=(0., .85), std_coeff=(.10, 0.)):
        super().__init__()
        self.register_buffer('mean_coeff', torch.tensor(mean_coeff, dtype=torch.float32))
        self.register_buffer('std_coeff', torch.tensor(std_coeff, dtype=torch.float32))

    @classmethod
    def fit(cls, transitions):
        pairs = [(float(t['state']['last_price']), float(t['action']['price']))
                 for t in transitions if t['action']['kind'] == 'offer' and t['action'].get('price') is not None]
        if len(pairs) < 3:
            raise ValueError('BRAC needs at least 3 logged seller offers to fit the behavior price prior.')
        arr = np.asarray(pairs, dtype=np.float64)
        x = np.column_stack([np.ones(len(arr)), arr[:, 0]])
        m = np.linalg.lstsq(x, arr[:, 1], rcond=None)[0]
        residual = np.abs(arr[:, 1] - x @ m)
        scale = np.linalg.lstsq(x, residual * np.sqrt(np.pi / 2), rcond=None)[0]
        # A floor prevents ill-defined KL on tiny interim datasets.
        return cls(tuple(m.astype(float)), tuple(scale.astype(float)))

    def distribution(self, last_price):
        mean = self.mean_coeff[0] + self.mean_coeff[1] * last_price
        std = (self.std_coeff[0] + self.std_coeff[1] * last_price).clamp(min=0.03, max=1.)
        return Normal(mean, std)


class PricePolicy(nn.Module):
    def __init__(self, state_feature_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_feature_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def distribution(self, state_features):
        out = self.net(state_features)
        # Squash mean to feasible normalized price range, constrain std.
        mean = .1 + 1.4 * torch.sigmoid(out[..., 0])
        std = (.03 + torch.nn.functional.softplus(out[..., 1])).clamp(max=.7)
        return Normal(mean, std)


def actor_loss(q_for_reparameterized_price, policy_dist, prior_dist, beta):
    """Minimize -Q(s,a') + beta*KL(pi || pi_B)."""
    return -q_for_reparameterized_price.mean() + beta * kl_divergence(policy_dist, prior_dist).mean()


def expected_target(reward, done, candidate_target_qs, gamma):
    """BRAC uses an expectation (Monte Carlo average), not proposal max."""
    nxt = candidate_target_qs.mean() if candidate_target_qs.numel() else torch.tensor(0.)
    return torch.as_tensor(reward, dtype=torch.float32, device=nxt.device) + (
        0 if done else gamma * nxt)
