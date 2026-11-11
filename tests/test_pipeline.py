import json
import random
import torch
import pytest
from data.fixture import make_fixture
from data.preprocess import prepare, read_jsonl, mask_prices, transitions_for_dialogue
from models.generator import TemplateGenerator, render
from models.q_network import QNetwork, soft_update
from rl.features import FeatureBuilder, HashedEncoder
from rl.prop import candidates, prop_target
from rl.cql import cql_regularizer
from rl.brac import PricePrior, PricePolicy, actor_loss, expected_target
from simulation.environment import RuleSeller, simulate
from training.train_q import train_q, QAgent
from config import seed_everything


def test_mask_and_action_candidates():
    assert mask_prices('Can you do $120 and pick up in 2026?') == 'Can you do [PRICE] and pick up in 2026?'
    assert mask_prices('I can do 85', 85) == 'I can do [PRICE]'
    assert mask_prices('iPhone 5S is good') == 'iPhone 5S is good'
    state = {'title': 'Desk', 'description': '', 'last_price': 1., 'history': [], 'last_type': 'message'}
    random.seed(42)
    actions = candidates(state, TemplateGenerator(), 5, 5)
    assert 1 <= len(actions) <= 25  # no redundant price copies for fixed responses
    assert len({(a['kind'], a['text'], a['price']) for a in actions}) == len(actions)
    for act in actions:
        if act['kind'] == 'offer':
            assert .7 <= act['price'] <= 1.0
        if act['kind'] == 'accept':
            assert render(act, 100) == 'I accept your offer.'


def test_fixture_parser_and_rewards(tmp_path):
    paths = make_fixture(tmp_path/'raw', count=10)
    prepare(paths, tmp_path/'processed')
    rows = read_jsonl(tmp_path/'processed'/'train_transitions.jsonl')
    assert len(rows) == 30
    successes = [r for r in rows if r['done'] and r['outcome'] == 'accept']
    failures = [r for r in rows if r['done'] and r['outcome'] == 'reject']
    assert successes and failures
    assert all(r['reward'] > 0 for r in successes)
    assert all(r['reward'] == -20 for r in failures)
    assert all(r['reward'] == 0 for r in rows if not r['done'])
    assert len({r['dialogue_id'] for r in rows}) == 10


def test_targets_and_cql():
    vals = torch.tensor([1., 2., 3.])
    assert torch.isclose(prop_target(2., False, vals, .9), torch.tensor(4.7))
    assert torch.isclose(expected_target(2., False, vals, .9), torch.tensor(3.8))
    assert torch.isclose(prop_target(-20., True, vals, .9), torch.tensor(-20.))
    log_q = torch.tensor(2.)
    assert torch.isclose(cql_regularizer(vals, log_q), torch.logsumexp(vals,0)-log_q)


def test_q_weights_soft_update():
    online = QNetwork(12)
    target = QNetwork(12)
    with torch.no_grad():
        for param in online.parameters():
            param.fill_(1.)
        for param in target.parameters():
            param.fill_(0.)
    soft_update(target, online, .05)
    assert torch.isclose(next(target.parameters()).flatten()[0], torch.tensor(.05))


def test_brac_actor_grad(tmp_path):
    prior = PricePrior()
    actor = PricePolicy(12)
    inp = torch.randn(1,12)
    dist = actor.distribution(inp)
    sampled = dist.rsample()
    q = sampled * 2  # differentiable score
    loss = actor_loss(q, dist, prior.distribution(torch.ones(1)), 1.)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters()) > 0


def test_training_and_checkpoint(tmp_path):
    seed_everything(42)
    paths = make_fixture(tmp_path/'raw', count=4)
    prepare(paths, tmp_path/'processed')
    infile = tmp_path/'processed'/'train_transitions.jsonl'
    for method in ('prop', 'cql', 'brac'):
        checkpoint = tmp_path/f'{method}.pt'
        train_q(method, TemplateGenerator(), encoder='hashed', episodes=1,
                train_file=infile, output=checkpoint, max_train=12, n_replies=2, n_prices=2)
        assert checkpoint.exists()
        agent = QAgent(checkpoint, TemplateGenerator())
        rows = read_jsonl(infile)
        action, scores = agent.decide(rows[0]['state'], 2, 2)
        assert 1 <= len(scores) <= 4
        assert len({(a['kind'], a['text'], a['price']) for a, _ in scores}) == len(scores)
        assert action['kind'] in ('offer', 'message', 'accept', 'reject')
        assert all(abs(score) < 1e6 for _, score in scores)


def test_simulation_terminal():
    outcome = simulate(RuleSeller(), 'cooperative', max_turns=6)
    assert outcome['turns'] <= 6
    assert outcome['deal'] == (outcome['final_price'] is not None)
    assert outcome['reward'] == (10 * outcome['final_price'] / 100 if outcome['deal'] else -20)
