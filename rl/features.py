"""State + action embeddings, scalar normalized prices, one-hot action types.

--encoder hashed is a deterministic CPU approximation to the paper's GPT-2
masked mean of hidden states. --encoder gpt2 uses the checkpoint to compute
masked mean last hidden states of full dialogue history, as in paper A.1.
"""
import hashlib
import re
from functools import lru_cache
import numpy as np
import torch
from models.generator import history_to_prompt

KINDS = ('message', 'offer', 'accept', 'reject')


class HashedEncoder:
    def __init__(self, dim=128):
        self.dim = dim

    @lru_cache(maxsize=20000)
    def encode(self, text):
        out = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r'[a-z0-9]+', text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], 'little') % self.dim
            out[index] += 1 if digest[4] % 2 else -1
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out


class GPT2Encoder:
    def __init__(self, generator):
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.device = generator.device
        self.dim = generator.hidden_size

    @lru_cache(maxsize=20000)
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.transformer(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            emb = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb[0].float().cpu().numpy()


class FeatureBuilder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.dim = 2 * encoder.dim + 10  # [s_embed, s_price, s_type(4), a_embed, a_price, a_type(4)]

    def state_features(self, state):
        return np.concatenate([self.encoder.encode(history_to_prompt(state)),
                 np.array([float(state.get('last_price') or 1)], dtype=np.float32),
                 self.one_hot(state.get('last_type', 'message'))])

    @staticmethod
    def one_hot(kind):
        v = np.zeros(4, dtype=np.float32)
        v[KINDS.index(kind if kind in KINDS else 'message')] = 1
        return v

    def action_features(self, state, action):
        text = action['text'] if action['kind'] == 'message' else action['kind']
        # Paper: action embedding uses full context+dialogue+candidate.
        prompt = history_to_prompt(state) + ' ' + text
        return np.concatenate([self.encoder.encode(prompt),
                 np.array([float(action.get('price') or 0)], dtype=np.float32),
                 self.one_hot(action['kind'])])

    def pair(self, state, action):
        return np.concatenate([self.state_features(state), self.action_features(state, action)])

    def pairs(self, state, actions):
        return torch.from_numpy(np.stack([self.pair(state, a) for a in actions]))
