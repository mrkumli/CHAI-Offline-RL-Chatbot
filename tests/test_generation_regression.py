"""Regression tests for GPT-2's EOS-only and repeated-control-word failures.

Mocks the text generation interface, not language model weights. Q policies are
loaded from temporary random-weight checkpoints, never trained here.
"""
import json

import pytest
import torch

from evaluation.evaluate import evaluate
from models.generator import GPT2Generator
from models.q_network import QNetwork
from rl.brac import PricePolicy
from rl.features import FeatureBuilder, HashedEncoder
from rl.prop import candidates
from simulation.environment import LMSeller, RuleSeller
from training.train_q import QAgent


STATE = {'title': 'Desk', 'description': '', 'last_price': .85,
         'last_type': 'offer', 'history': []}


class FakeTokenizer:
    eos_token = '[EOS]'
    eos_token_id = 99

    def __init__(self, values):
        self.values = values
        self.last_prompt = None

    def __call__(self, prompt, **_):
        self.last_prompt = prompt
        return {'input_ids': torch.tensor([[4, 5]]),
                'attention_mask': torch.ones(1, 2, dtype=torch.long)}

    def decode(self, generated, **_):
        return self.values[int(generated[0])]


class FakeModel:
    def __init__(self, outputs):
        # Each generate() call consumes one batch of strings. A single text can
        # also be used as a default reply for an arbitrary number of calls.
        self.outputs = outputs
        self.calls = 0
        self.params = []

    def generate(self, **kwargs):
        self.calls += 1
        self.params.append(kwargs)
        n = kwargs['num_return_sequences']
        batch = self.outputs.pop(0) if isinstance(self.outputs, list) else self.outputs
        if isinstance(batch, str):
            batch = [batch] * n
        assert len(batch) == n
        ids = [self.lookup[text] for text in batch]
        return torch.tensor([[4, 5, k] for k in ids])


def fake_gpt2(outputs):
    # Bypass real HF loading while testing the same GPT2Generator.sample method.
    generator = GPT2Generator.__new__(GPT2Generator)
    values = ['[EOS]', '\n', 'accept', 'offer', 'Would [PRICE] work?', 'Buyer: sorry']
    generator.tokenizer = FakeTokenizer(values)
    generator.model = FakeModel(outputs)
    generator.model.lookup = {v: i for i, v in enumerate(values)}
    generator.device = 'cpu'
    generator.torch = torch
    return generator


def test_empty_and_duplicate_generation_retries_and_aligns_prompt():
    gen = fake_gpt2([
        ['[EOS]', '\n', 'accept'],
        ['accept', 'offer'],
        ['Would [PRICE] work?'],
    ])
    replies = gen.sample(STATE, n=3)
    assert gen.model.calls == 3  # bounded retries
    assert gen.tokenizer.last_prompt.endswith('Seller: ')
    assert [r['text'] for r in replies] == ['accept', 'offer', 'Would [PRICE] work?']
    assert all(r['source'] == 'gpt2' for r in replies)
    assert gen.last_sampling_stats['empty'] == 2
    assert gen.last_sampling_stats['duplicates'] == 1
    assert gen.last_sampling_stats['fallback'] is False
    assert all(call['min_new_tokens'] >= 1 for call in gen.model.params)


def test_all_empty_generations_produce_visible_fallback_not_crash():
    gen = fake_gpt2('[EOS]')
    with pytest.warns(RuntimeWarning, match='explicitly flagged emergency offer'):
        replies = gen.sample(STATE, n=4)
    assert gen.model.calls == 3
    assert replies == [{'kind': 'offer', 'text': 'offer', 'source': 'gpt2_fallback'}]
    assert gen.last_sampling_stats['fallback'] is True
    actions = candidates(STATE, gen, n_replies=4, n_prices=3)
    assert len(actions) == 3  # price variation only for the marked offer
    assert all(action['source'] == 'gpt2_fallback' for action in actions)


def test_collapsed_accept_only_one_unique_q_action(tmp_path):
    gen = fake_gpt2('accept')
    with pytest.warns(RuntimeWarning, match='distinct replies'):
        actions = candidates(STATE, gen, n_replies=5, n_prices=5)
    assert gen.model.calls == 3
    assert actions == [{'kind': 'accept', 'text': 'accept', 'price': .85, 'source': 'gpt2'}]

    builder = FeatureBuilder(HashedEncoder())
    duplicate_vectors = builder.pairs(STATE, [actions[0]] * 25)
    assert torch.allclose(duplicate_vectors, duplicate_vectors[0].expand_as(duplicate_vectors))
    q = QNetwork(builder.dim).eval()
    with torch.no_grad():
        scores = q(duplicate_vectors)
    assert torch.allclose(scores, scores[0].expand_as(scores))

    varied = candidates(STATE, type('Generator', (), {
        'sample': lambda *_: [{'kind': 'offer', 'text': 'offer'}]})(),
        n_replies=1, n_prices=3, price_sampler=iter([.7, .8, .9]).__next__)
    assert len(varied) == 3
    price_vectors = builder.pairs(STATE, varied)
    assert not torch.allclose(price_vectors[0], price_vectors[1])


def test_full_evaluation_all_five_policies_with_fallback(tmp_path):
    gen = fake_gpt2('[EOS]')
    dim = FeatureBuilder(HashedEncoder()).dim
    policies = {'rule': RuleSeller(), 'lm': LMSeller(gen)}
    for method in ('prop', 'cql', 'brac'):
        checkpoint = tmp_path / f'q_{method}.pt'
        torch.save({'method': method, 'encoder': 'hashed', 'feature_dim': dim,
                    'q': QNetwork(dim).state_dict(),
                    'actor': PricePolicy(133).state_dict() if method == 'brac' else None},
                   checkpoint)
        policies[method] = QAgent(checkpoint, gen)
    path = tmp_path / 'evaluation.json'
    with pytest.warns(RuntimeWarning, match='emergency offer'):
        report = evaluate(policies, [{'title': 'Desk', 'list_price': 100.}],
                          episodes=1, output=path, n_replies=2, n_prices=3)
    assert set(report['policies']) == {'rule', 'lm', 'prop', 'cql', 'brac'}
    assert report['metadata']['n_replies'] == 2
    assert report['metadata']['n_prices'] == 3
    assert report['policies']['rule']['fallback_seller_turns'] == 0
    for name in ('lm', 'prop', 'cql', 'brac'):
        assert report['policies'][name]['fallback_seller_turns'] > 0
        assert 0 < report['policies'][name]['fallback_seller_turn_rate'] <= 1
    assert json.loads(path.read_text()) == report
