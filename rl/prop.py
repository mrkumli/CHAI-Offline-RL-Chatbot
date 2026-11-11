"""CHAI-prop: sample realistic candidate actions and maximize target Q over them."""
import random
import torch
from config import N_REPLIES, N_PRICES
from data.preprocess import PRICE_TOKEN


def price_proposals(previous_price, count=N_PRICES, sampler=None):
    if sampler is not None:
        return [float(sampler()) for _ in range(count)]
    # Paper Sec. 4.1: uniform in 70%-100% of previous offer.
    base = float(previous_price or 1)
    return [random.uniform(.70 * base, 1.0 * base) for _ in range(count)]


def candidates(state, generator, n_replies=N_REPLIES, n_prices=N_PRICES, price_sampler=None):
    """Make distinct (reply, price) actions; prices do not multiply control actions.

    For example, five identical 'accept' replies correspond to ONE action, not
    twenty-five different actions. Duplicate actions have identical Q features.
    """
    if n_replies < 1 or n_prices < 1:
        raise ValueError('n_replies and n_prices must both be >= 1.')
    replies = generator.sample(state, n_replies)
    if not replies:
        raise ValueError('Generator returned no replies; cannot construct actions.')
    prices = price_proposals(state.get('last_price', 1.0), n_prices, price_sampler)
    actions = []
    seen = set()
    for reply in replies:
        kind = reply['kind']
        text = reply['text']
        price_dependent = kind == 'offer' or PRICE_TOKEN in text
        # Do not create n_prices copies of 'accept', 'reject' or a fixed message.
        applicable_prices = prices if price_dependent else [state.get('last_price', 1.0) if kind == 'accept' else None]
        for price in applicable_prices:
            key = (kind, text.strip().casefold(), round(float(price), 8) if price is not None else None)
            if key in seen:
                continue
            seen.add(key)
            action = {'kind': kind, 'text': text, 'price': price}
            if 'source' in reply:
                action['source'] = reply['source']
            actions.append(action)
    return actions


def prop_target(reward, done, target_q_values, gamma):
    """r + gamma*(1-done)*max_i target Q(s',a'_i)."""
    next_q = target_q_values.max() if target_q_values.numel() else torch.tensor(0.0)
    return torch.as_tensor(reward, dtype=torch.float32, device=next_q.device) + (
        0 if done else gamma * next_q)
