"""Phase 2 tests: CraigslistBargain parsing, price masking, GPT-2 prompt handling.

Everything except the last test uses small hand-written records and FAKE
tokenizer/model objects. They check our Python logic (parsing, truncation,
duplicate removal, fallback reporting). They do NOT measure real GPT-2 quality.

The final test loads the REAL fine-tuned checkpoint if it exists
(checkpoints/generator, or the folder in env var CHAI_GENERATOR_DIR) and is
skipped otherwise.
"""
import json
import os

import pytest
import torch

from config import CHECKPOINTS, LM_LENGTH, LM_MAX_NEW_TOKENS, SUCCESS_MULTIPLIER
from data.preprocess import (PRICE_TOKEN, from_cocoa, has_malformed_price, mask_prices,
                             prepare, read_jsonl, transitions_for_dialogue)
from models.generator import GPT2Generator, history_to_prompt, response_type
from rl.prop import candidates


# ---------------------------------------------------------------- helpers

def cocoa_record(seller_first=False, offer_price=900.0, list_price=1000):
    """A tiny dialogue in the raw Cocoa format used by data/raw/*.json."""
    buyer = {'personal': {'Role': 'buyer', 'Target': 80},
             'item': {'Title': 'Bike', 'Price': list_price, 'Description': ['Red bike.', 'New tyres.']}}
    seller = {'personal': {'Role': 'seller', 'Target': 100},
              'item': {'Title': 'Bike', 'Price': list_price, 'Category': 'bike',
                       'Description': ['Red bike.', 'New tyres.']}}
    kbs = [seller, buyer] if seller_first else [buyer, seller]
    b, s = (1, 0) if seller_first else (0, 1)
    events = [
        {'agent': b, 'action': 'message', 'data': 'Would you take $850?', 'metadata': {'price': 850}},
        {'agent': s, 'action': 'message', 'data': 'It is listed at 1,000 but I can do 950', 'metadata': {'price': 950}},
        {'agent': b, 'action': 'quit', 'data': None, 'metadata': None},  # raw quit: no text
        {'agent': b, 'action': 'offer', 'data': {'price': offer_price, 'sides': ''}, 'metadata': None},
        {'agent': s, 'action': 'accept', 'data': None, 'metadata': None},
    ]
    return {'scenario': {'kbs': kbs}, 'events': events}


class FakeTokenizer:
    """Word-free fake: every prompt becomes the ids 0..length-1."""
    eos_token = '[EOS]'
    eos_token_id = 99

    def __init__(self, replies, prompt_length=5):
        self.replies = replies
        self.prompt_length = prompt_length
        self.last_prompt = None

    def __call__(self, prompt, **_):
        self.last_prompt = prompt
        ids = torch.arange(self.prompt_length).unsqueeze(0)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

    def decode(self, generated, **_):
        return self.replies[int(generated[0])]


class FakeModel:
    """Returns the scripted replies in order; records what generate() received."""
    def __init__(self, n_replies):
        self.next_reply = 0
        self.n_replies = n_replies
        self.seen_input_ids = []

    def generate(self, input_ids, num_return_sequences, **_):
        self.seen_input_ids.append(input_ids)
        rows = []
        for _ in range(num_return_sequences):
            reply_id = min(self.next_reply, self.n_replies - 1)
            self.next_reply += 1
            rows.append(input_ids[0].tolist() + [reply_id])
        return torch.tensor(rows)


def fake_generator(replies, prompt_length=5):
    generator = GPT2Generator.__new__(GPT2Generator)  # skip real HF loading
    generator.tokenizer = FakeTokenizer(replies, prompt_length)
    generator.model = FakeModel(len(replies))
    generator.device = 'cpu'
    generator.torch = torch
    return generator


STATE = {'title': 'Bike', 'description': 'Red bike.', 'last_price': .9, 'last_type': 'offer',
         'list_price': 100., 'history': [{'role': 'buyer', 'kind': 'offer', 'text': 'offer', 'price': .9}]}


# ---------------------------------------------------------------- dataset parsing

@pytest.mark.parametrize('seller_first', [False, True])
def test_roles_identified_from_kb_not_from_agent_order(seller_first):
    dialogue = from_cocoa(cocoa_record(seller_first=seller_first))
    roles = [e['role'] for e in dialogue['events']]
    assert roles == ['buyer', 'seller', 'buyer', 'seller']


def test_description_list_is_joined_and_quit_events_dropped():
    dialogue = from_cocoa(cocoa_record())
    assert dialogue['description'] == 'Red bike. New tyres.'  # not "['Red bike.', ...]"
    # The raw 'quit' (data=None) must not become an empty message.
    assert all(e['text'].strip() for e in dialogue['events'])
    assert [e['kind'] for e in dialogue['events']] == ['message', 'message', 'offer', 'accept']


def test_prices_normalized_by_list_price_and_reward_uses_final_offer():
    dialogue = from_cocoa(cocoa_record(offer_price=900.0))
    assert dialogue['list_price'] == 1000.0
    # '$850' is masked; '950' is masked because metadata confirms it; '1,000' is not.
    assert dialogue['events'][1]['text'] == f'It is listed at 1,000 but I can do {PRICE_TOKEN}'
    assert dialogue['events'][2]['price'] == pytest.approx(0.9)
    transitions = transitions_for_dialogue(dialogue, 'd0')
    # One transition per seller event: the message, then the terminal accept.
    assert [t['action']['kind'] for t in transitions] == ['message', 'accept']
    final = transitions[-1]
    assert final['done'] and final['outcome'] == 'accept'
    assert final['reward'] == pytest.approx(SUCCESS_MULTIPLIER * 0.9)
    # The state seen by the seller excludes its own action; next_state includes it.
    assert final['state']['history'][-1]['kind'] == 'offer'
    assert final['next_state']['history'][-1]['kind'] == 'accept'


def test_malformed_price_dialogues_are_skipped(tmp_path):
    good, bad = cocoa_record(offer_price=900.0), cocoa_record(offer_price=5000.0)  # 5x list price
    assert not has_malformed_price(from_cocoa(good)) and has_malformed_price(from_cocoa(bad))
    raw = {}
    for split in ('train', 'validation', 'test'):
        raw[split] = tmp_path / f'{split}.json'
        raw[split].write_text(json.dumps([good, bad]), encoding='utf-8')
    prepare(raw, tmp_path / 'out')
    for split in ('train', 'validation', 'test'):
        rows = read_jsonl(tmp_path / 'out' / f'{split}_dialogues.jsonl')
        assert len(rows) == 1 and rows[0]['id'] == f'{split}-0'
        # Each split is written from its own source file only (no cross-split mixing).
        assert all(t['dialogue_id'].startswith(split) for t in
                   read_jsonl(tmp_path / 'out' / f'{split}_transitions.jsonl'))


# ---------------------------------------------------------------- price masking

def test_mask_prices_dollar_comma_and_metadata_number():
    assert mask_prices('Would you take $1,000?') == f'Would you take {PRICE_TOKEN}?'
    assert mask_prices('listed at 2,800 today', 2800) == f'listed at {PRICE_TOKEN} today'
    assert mask_prices('only 2,800.00 firm', 2800) == f'only {PRICE_TOKEN} firm'
    assert mask_prices('I can do 95', 95) == f'I can do {PRICE_TOKEN}'


def test_mask_prices_leaves_unrelated_numbers():
    # Without metadata confirming it, a naked number is NOT assumed to be a price.
    assert mask_prices('Bought it in 2008, 48,000 miles', 2800) == 'Bought it in 2008, 48,000 miles'
    assert mask_prices('size 10 shoes') == 'size 10 shoes'


# ---------------------------------------------------------------- prompt construction

def test_prompt_format_matches_training():
    prompt = history_to_prompt(STATE)
    assert prompt.startswith('Item: Bike\nDescription: Red bike.')
    assert prompt.endswith(f'\nBuyer: offer {PRICE_TOKEN}\nSeller:')
    gen = fake_generator(['Sounds good.'])
    gen.sample(STATE, 1)
    # Training used history_to_prompt(state) + ' ' before the target text.
    assert gen.tokenizer.last_prompt == history_to_prompt(STATE) + ' '


def test_long_prompt_keeps_its_end_like_training():
    long_length = 400  # longer than LM_LENGTH=256
    gen = fake_generator(['Sounds good.'], prompt_length=long_length)
    gen.sample(STATE, 1)
    seen = gen.model.seen_input_ids[0][0].tolist()
    assert len(seen) == LM_LENGTH - LM_MAX_NEW_TOKENS
    # The LAST prompt tokens (newest turns + 'Seller: ') survive; the oldest are cut.
    assert seen[-1] == long_length - 1 and seen[0] == long_length - len(seen)


# ---------------------------------------------------------------- empty / duplicate / diversity

def test_empty_generations_retry_then_flagged_fallback():
    gen = fake_generator(['[EOS]', '\n', 'Buyer: hello'])  # never a usable seller reply
    with pytest.warns(RuntimeWarning, match='no usable reply'):
        replies = gen.sample(STATE, 3)
    assert replies == [{'kind': 'offer', 'text': 'offer', 'source': 'gpt2_fallback'}]
    stats = gen.last_sampling_stats
    assert stats['fallback'] and stats['unique'] == 0 and stats['empty'] == stats['attempted']


@pytest.mark.parametrize('text,kind', [
    ('accept', 'accept'), ('accept :)', 'accept'), ('acceptoffer', 'accept'),
    ('accept accept', 'accept'), ('Accepted!', 'accept'), ('accepting', 'accept'),
    ('accept Thank you for your offer', 'accept'), ('offer, good luck', 'offer'),
    ('accepts the price', 'accept'), ('accepted, see you soon', 'accept'),
    ('reject', 'reject'), ('offer [PRICE]', 'offer'),
    ('I accept your offer.', 'message'), ('Acceptable condition.', 'message'),
    ('Offer me [PRICE] and it is yours', 'message'), ('I can offer it for [PRICE].', 'message'),
])
def test_response_type_recognizes_control_word_variants(text, kind):
    assert response_type(text) == kind


def test_control_word_variants_count_as_one_duplicate_action():
    gen = fake_generator(['accept', 'accept :)', 'acceptoffer', 'Accepted!', 'accept Thank you'])
    with pytest.warns(RuntimeWarning, match='only 1/5 distinct'):
        replies = gen.sample(STATE, 5)
    assert replies == [{'kind': 'accept', 'text': 'accept', 'source': 'gpt2'}]
    assert gen.last_sampling_stats['duplicates'] >= 4


def test_diverse_replies_are_kept_and_become_distinct_actions():
    texts = ['Sounds good.', 'Would [PRICE] work?', 'It is almost new.', 'offer', 'accept']
    gen = fake_generator(texts)
    replies = gen.sample(STATE, 5)
    assert [r['text'] for r in replies] == texts and gen.last_sampling_stats['unique'] == 5
    gen = fake_generator(texts)
    actions = candidates(STATE, gen, n_replies=5, n_prices=3)
    # Price-dependent replies get 3 prices each; fixed text / accept get one action.
    assert len(actions) == 1 + 3 + 1 + 3 + 1
    keys = {(a['kind'], a['text'], a['price']) for a in actions}
    assert len(keys) == len(actions)


# ---------------------------------------------------------------- checkpoint loading

def test_missing_generator_checkpoint_gives_clear_error(tmp_path):
    pytest.importorskip('transformers')
    with pytest.raises(RuntimeError, match='Train with train-lm first'):
        GPT2Generator(tmp_path / 'does-not-exist')


REAL_GENERATOR = os.environ.get('CHAI_GENERATOR_DIR', str(CHECKPOINTS / 'generator'))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_GENERATOR, 'config.json')),
                    reason='real fine-tuned GPT-2 checkpoint not present')
def test_real_generator_checkpoint_loads_and_generates():
    pytest.importorskip('transformers')
    torch.manual_seed(0)
    gen = GPT2Generator(REAL_GENERATOR, device='cpu')
    tok = gen.tokenizer
    # [PRICE] is ONE added token and the embedding matrix was resized to match.
    price_ids = tok(PRICE_TOKEN, add_special_tokens=False)['input_ids']
    assert len(price_ids) == 1 and price_ids[0] == len(tok) - 1
    assert gen.model.get_input_embeddings().weight.shape[0] == len(tok)
    assert tok.pad_token == tok.eos_token
    # A long dialogue (>256 tokens) must still produce real, non-fallback replies.
    turns = [{'role': r, 'kind': 'message', 'text': 'I really like this bike and the price ' * 3,
              'price': None} for r in ('buyer', 'seller') * 6]
    state = dict(STATE, history=turns + STATE['history'])
    replies = gen.sample(state, 3)
    assert replies and all(r.get('source') == 'gpt2' for r in replies)
