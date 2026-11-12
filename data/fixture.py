"""Small INVENTED fixture for unit testing ONLY; not CraigslistBargain research data."""
import json
from pathlib import Path


def make_fixture(folder: Path, count=12):
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ('train', 'validation', 'test'):
        examples = []
        for i in range(count if split == 'train' else max(3, count // 3)):
            price = 100 + 10 * i
            close = price * (0.78 + (i % 4) * .04)
            reject = i % 5 == 0
            # Synthetic events mimic Stanford Cocoa's raw schema.
            examples.append({'scenario': {'kbs': [
                {'personal': {'Role': 'buyer', 'Target': close}, 'item': {'Title': f'Used desk {i}', 'Price': price}},
                {'personal': {'Role': 'seller', 'Target': price}, 'item': {'Title': f'Used desk {i}',
                  'Description': 'A sturdy wooden desk in good condition.', 'Category': 'furniture', 'Price': price}}]},
                'events': [
                    {'agent': 0, 'action': 'message', 'data': 'Is it available?'},
                    {'agent': 1, 'action': 'message', 'data': 'Yes, it is available for pickup.'},
                    {'agent': 0, 'action': 'offer', 'data': {'price': close}},
                    {'agent': 1, 'action': 'message', 'data': f'Can we agree at ${price * .95:.2f}?'},
                    {'agent': 0, 'action': 'message', 'data': 'Could you lower the price?'},
                    {'agent': 1, 'action': 'offer', 'data': {'price': round(close + 1, 2)}},
                    {'agent': 0, 'action': 'reject' if reject else 'accept', 'data': None}
                ]})
        path = folder / f'{split}.json'
        path.write_text(json.dumps(examples), encoding='utf-8')
        paths[split] = path
    return paths
