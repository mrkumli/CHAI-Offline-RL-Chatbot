"""Guards for Phase 1 cleanup: shared settings in config.py are actually used.

These tests protect behaviour that existing checkpoints depend on (layer sizes,
feature sizes) and the paths the CLI uses, so a later edit to config.py or to a
module cannot silently drift apart.
"""
import random
import config
from data.preprocess import ACTION_KINDS
from models.q_network import QNetwork
from rl.features import KINDS, HashedEncoder
from rl.prop import price_proposals


def test_q_network_uses_configured_hidden_size():
    # Saved q_*.pt checkpoints were trained with 256-wide layers.
    assert config.Q_HIDDEN == 256
    linear_layers = [m for m in QNetwork(10).net if hasattr(m, 'weight')]
    assert [tuple(m.weight.shape) for m in linear_layers] == [(256, 10), (256, 256), (1, 256)]


def test_hashed_encoder_default_dim_comes_from_config():
    assert HashedEncoder().dim == config.FEATURE_DIM == 128


def test_action_kinds_have_one_definition():
    # One-hot order must stay fixed: it is part of every saved feature vector.
    assert KINDS is ACTION_KINDS
    assert KINDS == ('message', 'offer', 'accept', 'reject')


def test_price_proposals_stay_in_configured_range():
    random.seed(0)
    prices = price_proposals(0.8, count=200)
    assert min(prices) >= config.PRICE_SAMPLE_LOW * 0.8
    assert max(prices) <= config.PRICE_SAMPLE_HIGH * 0.8


def test_data_paths_are_anchored_at_project_root():
    # Paths no longer depend on the folder the CLI is launched from.
    for path in (config.DATA, config.RAW_DATA, config.FIXTURE_RAW, config.FIXTURE_DATA):
        assert path.is_absolute() and config.ROOT in path.parents
    assert config.FIXTURE_DATA.name == 'fixture_processed'
    assert config.RAW_DATA.name == 'raw'
