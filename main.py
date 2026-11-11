"""CLI entrypoint. Run 'python main.py --help' for all demo commands."""
import argparse
from pathlib import Path
from config import DATA, CHECKPOINTS, RESULTS, N_REPLIES, N_PRICES, seed_everything
from data.preprocess import prepare, download_raw, read_jsonl
from data.fixture import make_fixture
from models.generator import GPT2Generator, TemplateGenerator
from training.train_generator import train_generator
from training.train_q import train_q, QAgent
from simulation.environment import RuleSeller, LMSeller, simulate
from evaluation.evaluate import evaluate


def get_generator(args):
    return TemplateGenerator() if args.backend == 'template' else GPT2Generator(args.generator)


def main():
    parser = argparse.ArgumentParser(description='CHAI offline-RL research prototype')
    subs = parser.add_subparsers(dest='cmd', required=True)
    prep = subs.add_parser('prepare', help='Create normalized data and seller RL transitions')
    prep.add_argument('--download', action='store_true', help='Download official CraigslistBargain source')
    prep.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    prep.add_argument('--fixture', action='store_true', help='Synthetic toy data ONLY for smoke tests')
    prep.add_argument('--limit', type=int, default=0)
    prep.add_argument('--output', type=Path, default=DATA)
    lm = subs.add_parser('train-lm', help='Fine-tune GPT-2 on seller response text')
    lm.add_argument('--model', default='gpt2')
    lm.add_argument('--epochs', type=int, default=1)
    lm.add_argument('--batch-size', type=int, default=2)
    lm.add_argument('--max-train', type=int, default=0)
    lm.add_argument('--data-dir', type=Path, default=DATA)
    for name in ('train-q', 'demo', 'evaluate'):
        sub = subs.add_parser(name)
        sub.add_argument('--backend', choices=['gpt2', 'template'], default='gpt2')
        sub.add_argument('--data-dir', type=Path, default=DATA)
        sub.add_argument('--generator', type=Path, default=CHECKPOINTS/'generator')
        sub.add_argument('--n-replies', type=int, default=N_REPLIES)
        sub.add_argument('--n-prices', type=int, default=N_PRICES)
        if name in ('demo', 'evaluate'):
            sub.add_argument('--checkpoints', type=Path, default=None, help='Folder with q_{method}.pt')
        if name == 'train-q':
            sub.add_argument('--method', choices=['prop', 'cql', 'brac'], required=True)
            sub.add_argument('--encoder', choices=['hashed', 'gpt2'], default='hashed')
            sub.add_argument('--epochs', type=int, default=3)
            sub.add_argument('--max-train', type=int, default=0)
        elif name == 'demo':
            sub.add_argument('--policy', choices=['rule', 'lm', 'prop', 'cql', 'brac'], default='prop')
            sub.add_argument('--buyer', choices=['cooperative', 'stingy', 'sensitive'], default='cooperative')
        else:
            sub.add_argument('--policies', default='rule,lm,prop,cql,brac')
            sub.add_argument('--episodes', type=int, default=5)
            sub.add_argument('--label', default='real', choices=['real', 'fixture'])
    subs.add_parser('smoke', help='Run fixture + 3 Q methods + evaluation, no GPT-2 downloads')
    args = parser.parse_args()
    seed_everything()
    if args.cmd == 'prepare':
        raw = make_fixture(Path('data/fixture_raw')) if args.fixture else (
            download_raw(args.raw_dir) if args.download else
            {s: args.raw_dir/f'{s}.json' for s in ('train', 'validation', 'test')})
        output_dir = Path('data/fixture_processed') if args.fixture and args.output == DATA else args.output
        prepare(raw, output_dir, args.limit or None)
        print('Example:', read_jsonl(output_dir/'train_transitions.jsonl')[0])
        print('SOURCE:', 'SYNTHETIC TEST FIXTURE' if args.fixture else 'ORIGINAL CRAIGSLISTBARGAIN')
    elif args.cmd == 'train-lm':
        train_generator(name=args.model, epochs=args.epochs,
                        train_file=args.data_dir/'train_transitions.jsonl',
                        val_file=args.data_dir/'validation_transitions.jsonl',
                        batch_size=args.batch_size, max_train=args.max_train)
    elif args.cmd == 'train-q':
        q_save = (CHECKPOINTS/'fixture'/f'q_{args.method}.pt') if args.backend == 'template' else None
        train_q(args.method, get_generator(args), encoder=args.encoder, episodes=args.epochs,
                output=q_save, train_file=args.data_dir/'train_transitions.jsonl',
                max_train=args.max_train, n_replies=args.n_replies, n_prices=args.n_prices)
    elif args.cmd == 'smoke':
        print('IMPORTANT: SYNTHETIC SMOKE TEST. NOT CRAIGSLISTBARGAIN RESULTS.')
        raw = make_fixture(Path('data/fixture_raw'))
        fixture_data = Path('data/fixture_processed')
        prepare(raw, fixture_data)
        for method in ('prop', 'cql', 'brac'):
            train_q(method, TemplateGenerator(), episodes=1, max_train=12, n_replies=3, n_prices=3,
                    output=CHECKPOINTS/'fixture'/f'q_{method}.pt',
                    train_file=fixture_data/'train_transitions.jsonl')
        dialogues = read_jsonl(fixture_data/'test_dialogues.jsonl')
        policies = {'rule': RuleSeller(), 'lm-template': LMSeller(TemplateGenerator())}
        policies.update({m: QAgent(CHECKPOINTS/'fixture'/f'q_{m}.pt', TemplateGenerator())
                         for m in ('prop', 'cql', 'brac')})
        evaluate(policies, dialogues, episodes=2, output=RESULTS/'fixture_smoke.json')
    else:
        gen = get_generator(args)
        ckpt_dir = args.checkpoints or (CHECKPOINTS/'fixture' if args.backend == 'template' else CHECKPOINTS)
        if args.cmd == 'demo':
            if args.policy == 'rule':
                seller = RuleSeller()
            elif args.policy == 'lm':
                seller = LMSeller(gen)
            else:
                seller = QAgent(ckpt_dir/f'q_{args.policy}.pt', gen)
            rows = read_jsonl(args.data_dir/'test_dialogues.jsonl')
            d = rows[0]
            print('Scenario:', d['title'], '| list price:', d['list_price'])
            if args.backend == 'template':
                print('WARNING: template backend is NOT fine-tuned GPT-2.')
            simulate(seller, args.buyer, title=d['title'], list_price=d['list_price'],
                     verbose=True, show_q=True,
                     n_replies=args.n_replies, n_prices=args.n_prices)
        else:
            selection = [p.strip() for p in args.policies.split(',') if p.strip()]
            policies = {}
            for name in selection:
                if name == 'rule':
                    policies[name] = RuleSeller()
                elif name == 'lm':
                    policies[name] = LMSeller(gen)
                elif name in ('prop', 'cql', 'brac'):
                    policies[name] = QAgent(ckpt_dir/f'q_{name}.pt', gen)
                else:
                    parser.error(f'Invalid policy: {name}')
            path = RESULTS/('fixture_evaluation.json' if args.label == 'fixture' else 'evaluation.json')
            evaluate(policies, read_jsonl(args.data_dir/'test_dialogues.jsonl'), args.episodes, path,
                     n_replies=args.n_replies, n_prices=args.n_prices)
            print('DATA LABEL:', args.label.upper(), '(user-provided; verify source before reporting)')


if __name__ == '__main__':
    main()
