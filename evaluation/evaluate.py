"""Paired seed-controlled simulated evaluation; does not impersonate human study."""
import json
import random
from pathlib import Path
from config import RESULTS, SEED, N_REPLIES, N_PRICES, seed_everything
from data.preprocess import read_jsonl
from simulation.environment import simulate


def evaluate(policies, dialogues, episodes=5, output=RESULTS/'evaluation.json',
             n_replies=N_REPLIES, n_prices=N_PRICES):
    if not dialogues:
        raise ValueError('No test dialogues: run prepare first.')
    if episodes < 1:
        raise ValueError('episodes must be >= 1.')
    scenarios = []
    for style in ('cooperative', 'stingy', 'sensitive'):
        for i in range(episodes):
            d = dialogues[i % len(dialogues)]
            scenarios.append((style, d, SEED + i + 100 * len(style)))
    result = {'metadata': {'episodes_per_buyer': episodes,
                           'buyer_types': ['cooperative', 'stingy', 'sensitive'],
                           'n_replies': n_replies, 'n_prices': n_prices,
                           'fallback_note': 'gpt2_fallback means GPT-2 generated no usable text after bounded retries; these actions are NOT GPT-2 outputs',
                           'note': 'Hand-coded buyer simulator; NOT CHAI paper Table 1'}, 'policies': {}}
    for label, seller in policies.items():
        records = []
        all_replies = []
        for style, dialogue, seed in scenarios:
            seed_everything(seed)
            run = simulate(seller, style, title=dialogue['title'], list_price=dialogue['list_price'],
                           n_replies=n_replies, n_prices=n_prices)
            records.append({k: run[k] for k in ('deal', 'final_price', 'reward', 'turns', 'style',
                                                 'fallback_seller_turns')})
            all_replies.extend(line.strip().lower() for role, line in run['transcript'] if role == 'seller')
        n = len(records)
        accepted = [r for r in records if r['deal']]
        # Revenue counts failed negotiations as 0; normalized by list price.
        revs = [r['final_price'] / scenarios[i][1]['list_price'] if r['deal'] else 0
                for i, r in enumerate(records)]
        report = {'deal_success_rate': sum(r['deal'] for r in records) / n,
                  'avg_normalized_revenue_all': sum(revs) / n,
                  'avg_price_on_success': sum(r['final_price'] for r in accepted) / len(accepted) if accepted else None,
                  'avg_seller_reward': sum(r['reward'] for r in records) / n,
                  'avg_turns': sum(r['turns'] for r in records) / n,
                  'fallback_seller_turns': sum(r['fallback_seller_turns'] for r in records),
                  'fallback_seller_turn_rate': (sum(r['fallback_seller_turns'] for r in records) /
                                                sum(r['turns'] for r in records)),
                  'nonempty_reply_rate': sum(bool(r) for r in all_replies) / len(all_replies),
                  'distinct_reply_fraction': len(set(all_replies)) / len(all_replies),
                  'by_buyer': {style: sum(r['deal'] for r in records if r['style'] == style)/episodes
                               for style in ('cooperative', 'stingy', 'sensitive')}}
        result['policies'][label] = report
        print(f'{label:10} success={report["deal_success_rate"]:.1%} '
              f'revenue={report["avg_normalized_revenue_all"]:.3f} '
              f'reward={report["avg_seller_reward"]:+.2f} turns={report["avg_turns"]:.2f} '
              f'fallback_turns={report["fallback_seller_turns"]}')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Metrics saved:', path)
    return result
