"""CraigslistBargain raw Cocoa JSON -> normalized seller-only RL transitions.

Reads official train/dev/test separately to prevent dialogue leakage. Does not
infer test labels when the released test annotations are absent.
"""
import json
import re
import urllib.request
from pathlib import Path
from typing import Any
from config import DATA, REJECTION_REWARD, SUCCESS_MULTIPLIER

# These are the original data URLs used by the Stanford dataset loading script.
SOURCES = {
    'train': 'https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json',
    'validation': 'https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json',
    'test': 'https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/',
}
PRICE_TOKEN = '[PRICE]'
MONEY_RE = re.compile(r'(?<!\w)\$\s*\d[\d,]*(?:\.\d+)?')


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. First run: python main.py prepare --download')
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def clean_price(number):
    try:
        value = float(str(number).replace('$', '').replace(',', ''))
        return value if value > 0 else None
    except (ValueError, TypeError):
        return None


def mask_prices(text: str, offered_price=None):
    """Mask explicit currency, plus a naked number ONLY when metadata confirms it."""
    text = MONEY_RE.sub(PRICE_TOKEN, str(text))
    if offered_price is not None:
        value = float(offered_price)
        if value.is_integer():
            variants = [str(int(value)), f'{value:.2f}']
        else:
            variants = [str(value), f'{value:.2f}']
        for variant in variants:
            text = re.sub(r'(?<![\w.])' + re.escape(variant) + r'(?![\w.])', PRICE_TOKEN, text)
    return text.strip()


def normalized_event(kind, text, price, list_price):
    # A message may contain an explicit $ amount even when rule-based
    # annotations are missing; keep its numeric value for later substitution.
    if price is None and isinstance(text, str):
        amounts = MONEY_RE.findall(text)
        if len(amounts) == 1:
            price = clean_price(amounts[0])
    kind = kind.lower().strip()
    if kind not in ('message', 'offer', 'accept', 'reject'):
        kind = 'message'
    price = clean_price(price)
    if kind == 'offer':
        text = 'offer'
    elif kind in ('accept', 'reject'):
        text = kind
    else:
        text = mask_prices(text, price)
    return {'kind': kind, 'text': text, 'price': round(price / list_price, 5) if price else None}


def from_cocoa(raw):
    """Cocoa source JSON: scenario.kbs, events[].{agent,action,data,metadata}."""
    kbs = raw['scenario']['kbs']
    roles = [str(k['personal'].get('Role', '')).lower() for k in kbs]
    seller_id = roles.index('seller')
    item = kbs[seller_id]['item']
    list_price = clean_price(item.get('Price'))
    if not list_price:
        return None
    events = []
    for e in raw.get('events', []):
        if not isinstance(e.get('agent'), int) or e['agent'] not in (0, 1):
            continue
        meta = e.get('metadata') or {}
        kind = str(e.get('action', 'message')).lower()
        if kind not in ('offer', 'accept', 'reject'):
            kind = 'message'
        value = e.get('data')
        if isinstance(value, dict):
            extracted = clean_price(value.get('price'))
            text = str(value.get('text', ''))
        else:
            extracted = clean_price(value) if kind == 'offer' else None
            text = value if isinstance(value, str) else ''
        # Metadata is annotated by a rule-based model; prefer explicit offer event price.
        annotated = clean_price(meta.get('price'))
        price = extracted if extracted is not None else annotated
        if kind == 'offer' and not price:
            continue
        event = normalized_event(kind, text, price, list_price)
        event['role'] = 'seller' if e['agent'] == seller_id else 'buyer'
        events.append(event)
    return {'title': str(item.get('Title', '')), 'description': str(item.get('Description', ''))[:500],
            'category': str(item.get('Category', '')), 'list_price': list_price, 'events': events}


def from_hf(row):
    """Also supports manually exported Hugging Face features in JSON rows."""
    info = row['agent_info']
    roles = info['Role'] if isinstance(info, dict) else [x['Role'] for x in info]
    seller_id = [str(x).lower() for x in roles].index('seller')
    items = row['items']
    item = {key: (values[seller_id] if isinstance(values, list) else values)
            for key, values in items.items()} if isinstance(items, dict) else items[seller_id]
    price = clean_price(item.get('Price'))
    if not price:
        return None
    acts = row.get('dialogue_acts', {})
    intentions = acts.get('intent', []) if isinstance(acts, dict) else [a.get('intent', '') for a in acts]
    prices = acts.get('price', []) if isinstance(acts, dict) else [a.get('price') for a in acts]
    events = []
    for i, (agent, text) in enumerate(zip(row['agent_turn'], row['utterance'])):
        if agent not in (0, 1):
            continue
        intent = str(intentions[i]).lower() if i < len(intentions) else ''
        kind = intent if intent in ('offer', 'accept', 'reject') else 'message'
        offered = clean_price(prices[i]) if i < len(prices) else None
        if kind == 'offer' and not offered:
            continue
        event = normalized_event(kind, text, offered, price)
        event['role'] = 'seller' if agent == seller_id else 'buyer'
        events.append(event)
    return {'title': str(item.get('Title', '')), 'description': str(item.get('Description', ''))[:500],
            'category': str(item.get('Category', '')), 'list_price': price, 'events': events}


def get_dataset(filename: Path):
    raw = json.loads(filename.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError(f'Expected a JSON array in {filename}')
    return [e for row in raw if (e := (from_cocoa(row) if 'scenario' in row else from_hf(row))) is not None]


def transitions_for_dialogue(dialogue, dialogue_id):
    events = dialogue['events']
    history = []
    last_price = 1.0
    last_type = 'message'
    last_offer = None
    result = []
    for i, event in enumerate(events):
        if event['kind'] == 'offer' and event['price'] is not None:
            last_offer = event['price']
        if event['role'] != 'seller':
            history.append(event)
            if event['price'] is not None:
                last_price = event['price']
            last_type = event['kind']
            continue
        state = {'title': dialogue['title'], 'description': dialogue['description'],
                 'list_price': dialogue['list_price'], 'history': list(history),
                 'last_price': last_price, 'last_type': last_type}
        # One seller action then all buyer events before next seller action.
        suffix = []
        for next_event in events[i + 1:]:
            if next_event['role'] == 'seller':
                break
            suffix.append(next_event)
        next_history = history + [event] + suffix
        final = event if event['kind'] in ('accept', 'reject') else next(
            (e for e in suffix if e['kind'] in ('accept', 'reject')), None)
        done = bool(final) or not any(e['role'] == 'seller' for e in events[i + 1:])
        reward = 0.0
        if final and final['kind'] == 'reject':
            reward = REJECTION_REWARD
        elif final and final['kind'] == 'accept' and last_offer is not None:
            reward = SUCCESS_MULTIPLIER * last_offer
        next_price = next((e['price'] for e in reversed(next_history) if e['price'] is not None), last_price)
        next_type = next_history[-1]['kind'] if next_history else 'message'
        next_state = {**state, 'history': next_history, 'last_price': next_price,
                      'last_type': next_type}
        result.append({'dialogue_id': dialogue_id, 'state': state,
                       'action': {key: event[key] for key in ('kind', 'text', 'price')},
                       'reward': float(reward), 'next_state': next_state, 'done': done,
                       'outcome': final['kind'] if final else 'unknown'})
        # Stop adding actions at explicit terminal; later events may be metadata cleanup.
        history.append(event)
        if event['price'] is not None:
            last_price = event['price']
        last_type = event['kind']
        if final:
            break
    return result


def prepare(sources, output=DATA, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation', 'test'):
        path = sources[split]
        dialogues = get_dataset(path)
        if limit:
            dialogues = dialogues[:limit]
        for i, dialogue in enumerate(dialogues):
            dialogue['id'] = f'{split}-{i}'
        transitions = [t for d in dialogues for t in transitions_for_dialogue(d, d['id'])]
        write_jsonl(output / f'{split}_dialogues.jsonl', dialogues)
        write_jsonl(output / f'{split}_transitions.jsonl', transitions)
        print(f'{split}: {len(dialogues)} dialogues, {len(transitions)} seller transitions')
    return output


def download_raw(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, url in SOURCES.items():
        target = destination / f'{split}.json'
        if not target.exists():
            try:
                print('Downloading', split, '...', flush=True)
                urllib.request.urlretrieve(url, target)
            except Exception as error:
                target.unlink(missing_ok=True)
                raise RuntimeError(f'Failed to download {split}. Download the raw CraigslistBargain '
                                   f'file manually and save it as {target}. URL: {url}. '
                                   f'Original error: {error}') from error
        result[split] = target
    return result
