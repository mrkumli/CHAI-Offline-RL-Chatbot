"""Independent CHAI-prop, CHAI-CQL and CHAI-BRAC critic training.

No environment interactions during training. For the Q bootstrap, the proposal
set is derived from a fixed language model (or an explicitly labeled fixture
replacement), not the simulator.
"""
import json
import random
from pathlib import Path
import torch
from config import (DATA, CHECKPOINTS, Q_LR, GAMMA, TARGET_TAU, CQL_ALPHA,
                    BRAC_BETA, seed_everything, device)
from data.preprocess import read_jsonl
from models.q_network import QNetwork, soft_update
from rl.features import FeatureBuilder, HashedEncoder, GPT2Encoder
from rl.prop import candidates, prop_target
from rl.cql import cql_loss
from rl.brac import PricePrior, PricePolicy, actor_loss, expected_target


def make_features(encoder, generator=None):
    if encoder == 'hashed':
        return FeatureBuilder(HashedEncoder())
    if encoder == 'gpt2' and generator is not None and hasattr(generator, 'model'):
        return FeatureBuilder(GPT2Encoder(generator))
    raise ValueError('--encoder gpt2 requires a fine-tuned GPT-2 generator checkpoint.')


def train_q(method, generator, encoder='hashed', episodes=3, max_train=0,
            output=None, train_file=DATA/'train_transitions.jsonl', n_replies=5, n_prices=5):
    if method not in ('prop', 'cql', 'brac'):
        raise ValueError('Method must be prop, cql, or brac.')
    seed_everything()
    rows = read_jsonl(Path(train_file))
    if max_train:
        rows = rows[:max_train]
    if not rows:
        raise ValueError('No offline transitions available.')
    features = make_features(encoder, generator)
    dev = device()
    online = QNetwork(features.dim).to(dev)
    target = QNetwork(features.dim).to(dev)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=Q_LR)
    prior = PricePrior.fit(rows).to(dev) if method == 'brac' else None
    actor = PricePolicy(features.encoder.dim + 5).to(dev) if method == 'brac' else None
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=Q_LR) if actor else None

    # Cache fixed LM proposals. They are independent from Q learning updates.
    proposals = {}
    current_proposals = {}
    for i, row in enumerate(rows):
        if not row['done']:
            proposals[i] = candidates(row['next_state'], generator, n_replies, n_prices)
    if method == 'cql':
        for i, row in enumerate(rows):
            current_proposals[i] = candidates(row['state'], generator, n_replies, n_prices)
    # All methods start updates from the same random state and replay order.
    # In particular, extra CQL negative sampling must not alter prop proposals.
    seed_everything()
    for epoch in range(episodes):
        order = list(range(len(rows)))
        random.shuffle(order)
        total = 0.
        for i in order:
            row = rows[i]
            logged = features.pairs(row['state'], [row['action']]).to(dev)
            online_val = online(logged).squeeze()
            if not row['done']:
                if method == 'brac':
                    # Same LM text proposals, prices sampled from learned price policy.
                    with torch.no_grad():
                        state_vec = torch.tensor(features.state_features(row['next_state']), device=dev).unsqueeze(0)
                        sampled = actor.distribution(state_vec).sample((len(proposals[i]),)).flatten().clamp(.1, 1.5)
                    next_actions = [dict(a, price=float(sampled[j])) if a['kind'] == 'offer'
                                    or '[PRICE]' in a['text'] else a for j, a in enumerate(proposals[i])]
                else:
                    next_actions = proposals[i]
                with torch.no_grad():
                    target_vals = target(features.pairs(row['next_state'], next_actions).to(dev))
                    next_target = (expected_target if method == 'brac' else prop_target)(
                        row['reward'], False, target_vals, GAMMA)
            else:
                next_target = torch.tensor(row['reward'], dtype=torch.float32, device=dev)
            loss = (online_val - next_target).square()
            if method == 'cql':
                candidate_actions = current_proposals[i]
                negatives = online(features.pairs(row['state'], candidate_actions).to(dev))
                loss = cql_loss(loss, negatives, online_val, CQL_ALPHA)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 5.)
            optimizer.step()
            soft_update(target, online, TARGET_TAU)
            total += float(loss.detach())
            if method == 'brac' and row['action']['kind'] == 'offer':
                # Keep Q frozen during actor update, but differentiate Q wrt price.
                state_vec = torch.tensor(features.state_features(row['state']), device=dev).unsqueeze(0)
                dist = actor.distribution(state_vec)
                sampled_price = dist.rsample().clamp(.1, 1.5)
                q_in = features.pairs(row['state'], [row['action']]).to(dev).detach()
                # Action price is immediately after the action text embedding.
                price_index = 2 * features.encoder.dim + 5
                q_in[:, price_index] = sampled_price
                with torch.no_grad():
                    prior_dist = prior.distribution(state_vec[:, features.encoder.dim])
                for p in online.parameters():
                    p.requires_grad_(False)
                q_sample = online(q_in)
                a_loss = actor_loss(q_sample, dist, prior_dist, BRAC_BETA)
                actor_optimizer.zero_grad()
                a_loss.backward()
                actor_optimizer.step()
                for p in online.parameters():
                    p.requires_grad_(True)
        print(f'Q-{method} epoch {epoch+1}/{episodes}: avg_loss={total/len(rows):.4f}, transitions={len(rows)}')
    output = Path(output or CHECKPOINTS/f'q_{method}.pt')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'method': method, 'encoder': encoder, 'feature_dim': features.dim,
                'q': online.state_dict(), 'actor': actor.state_dict() if actor else None,
                'prior': prior.state_dict() if prior else None,
                'train_examples': len(rows)}, output)
    print('Saved critic:', output)
    return output


class QAgent:
    def __init__(self, checkpoint, generator):
        try:
            obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Missing Q checkpoint {checkpoint}. Run train-q first.') from exc
        self.method = obj['method']
        self.generator = generator
        self.features = make_features(obj['encoder'], generator)
        if self.features.dim != obj['feature_dim']:
            raise ValueError('Encoder dimensions changed since Q checkpoint was saved.')
        self.q = QNetwork(self.features.dim)
        self.q.load_state_dict(obj['q'])
        self.q.eval()
        self.actor = None
        if self.method == 'brac':
            self.actor = PricePolicy(self.features.encoder.dim + 5)
            self.actor.load_state_dict(obj['actor'])
            self.actor.eval()

    def decide(self, state, n_replies=5, n_prices=5, show=False):
        price_sampler = None
        if self.actor:
            state_vec = torch.from_numpy(self.features.state_features(state)).unsqueeze(0)
            def price_sampler():
                with torch.no_grad():
                    return float(self.actor.distribution(state_vec).sample().clamp(.1, 1.5)[0])
        opts = candidates(state, self.generator, n_replies, n_prices, price_sampler)
        with torch.no_grad():
            values = self.q(self.features.pairs(state, opts))
            # Paper softmax selection rather than unconditional argmax.
            selected = int(torch.multinomial(torch.softmax(values, 0), 1).item())
        if show:
            print(f'Unique candidate actions: {len(opts)} (requested up to {n_replies} replies x {n_prices} prices).')
            distinct_q = len({round(float(q), 6) for q in values})
            if distinct_q == 1:
                print('NOTE: All Q-values are equal at 6 decimals. If only one action exists, this is expected; '
                      'otherwise inspect Q sensitivity/checkpoint training.')
            print('Candidate actions and Q-values:')
            for i, (action, q) in enumerate(zip(opts, values)):
                print(f'  {i:02d} Q={float(q):+10.6f}  {action["kind"]:7s}  '
                      f'price={action.get("price")}  {action["text"][:50]}')
            print('Selected candidate:', selected)
        return opts[selected], [(a, float(q)) for a, q in zip(opts, values)]
