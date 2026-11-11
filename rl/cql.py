"""CQL(H) regularizer from CHAI section 4.1, approximating action sum by proposals."""
import torch


def cql_regularizer(online_proposals: torch.Tensor, online_data: torch.Tensor):
    # logsumexp over sampled proposal actions - Q value at logged action.
    # The continuous/infinite full action space is approximated with samples.
    return torch.logsumexp(online_proposals, dim=0) - online_data


def cql_loss(bellman_mse, proposal_qs, logged_q, alpha):
    return bellman_mse + alpha * cql_regularizer(proposal_qs, logged_q)
