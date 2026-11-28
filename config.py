"""Small, explicit defaults for reproducible interim experiments."""
from pathlib import Path
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'processed'
RAW_DATA = ROOT / 'data' / 'raw'  # official CraigslistBargain source JSON
FIXTURE_RAW = ROOT / 'data' / 'fixture_raw'  # synthetic smoke-test source JSON
FIXTURE_DATA = ROOT / 'data' / 'fixture_processed'  # synthetic smoke-test transitions
CHECKPOINTS = ROOT / 'checkpoints'
RESULTS = ROOT / 'results'
SEED = 42
LM_NAME = 'gpt2'  # original paper used gpt2-medium
LM_LENGTH = 256
LM_EPOCHS = 1
LM_BATCH = 2
LM_LR = 5e-5
LM_SAMPLE_ATTEMPTS = 3  # bounded retries for empty/duplicate GPT-2 replies
LM_MAX_NEW_TOKENS = 32  # reply length limit; prompt keeps its last LM_LENGTH - 32 tokens
MAX_PRICE_RATIO = 3.0  # prices above 3x list price are treated as malformed records
N_REPLIES = 5
N_PRICES = 5
PRICE_SAMPLE_LOW = 0.70  # paper Sec. 4.1: sample 70%-100% of the previous price
PRICE_SAMPLE_HIGH = 1.00
FEATURE_DIM = 128  # hashed fallback; GPT-2 encoder uses model hidden size
Q_HIDDEN = 256  # paper hidden size
Q_LR = 3e-4  # paper A.2
GAMMA = 0.99  # chosen for interim prototype
TARGET_TAU = 0.05  # paper A.2
CQL_ALPHA = 1.0  # chosen for interim prototype
BRAC_BETA = 1.0  # chosen for interim prototype
BRAC_PRICE_MIN = 0.1  # BRAC sampled prices are clipped to [min, max] x list price
BRAC_PRICE_MAX = 1.5
REJECTION_REWARD = -20.0  # paper, Section 3.1
SUCCESS_MULTIPLIER = 10.0  # paper, Section 3.1


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
