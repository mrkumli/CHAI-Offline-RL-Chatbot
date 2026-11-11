"""GPT-2 dialogue model; TemplateGenerator exists only for offline smoke tests."""
import random
import warnings
from config import LM_LENGTH, LM_SAMPLE_ATTEMPTS
from data.preprocess import PRICE_TOKEN


def history_to_prompt(state):
    title = str(state.get('title', ''))[:120]
    desc = str(state.get('description', ''))[:220]
    lines = [f'Item: {title}', f'Description: {desc}']
    for turn in state.get('history', [])[-12:]:
        speaker = 'Seller' if turn['role'] == 'seller' else 'Buyer'
        # Price is expressed as its own state variable; text includes placeholders.
        txt = turn['kind'] if turn['kind'] != 'message' else turn['text']
        if turn['kind'] == 'offer' and turn.get('price') is not None:
            txt = f"offer {PRICE_TOKEN}"
        lines.append(f'{speaker}: {txt}')
    return '\n'.join(lines) + '\nSeller:'


def response_type(text):
    head = text.strip().lower().strip(' .!')
    # Entire single-word commands only. Avoid "I accept..." being mistaken for a control action.
    if head == 'accept':
        return 'accept'
    if head == 'reject':
        return 'reject'
    if head.startswith('offer') and len(head.split()) <= 2:
        return 'offer'
    return 'message'


def render(action, list_price):
    price = float(action.get('price') or 1) * list_price
    kind = action['kind']
    if kind == 'offer':
        return f'I offer ${price:.2f}.'
    if kind == 'accept':
        return 'I accept your offer.'
    if kind == 'reject':
        return 'I cannot accept that offer.'
    return action['text'].replace(PRICE_TOKEN, f'${price:.2f}')


class GPT2Generator:
    def __init__(self, checkpoint, device=None):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError('GPT-2 needs transformers. Run pip install -r requirements.txt') from exc
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
            self.model = AutoModelForCausalLM.from_pretrained(str(checkpoint)).to(self.device).eval()
        except OSError as exc:
            raise RuntimeError(f'Missing GPT-2 model/checkpoint {checkpoint}. Train with train-lm first.') from exc
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def hidden_size(self):
        return self.model.config.n_embd

    def sample(self, state, n=5):
        """Return distinct GPT-2 replies, retrying empty/duplicate samples a bounded number of times.

        If *every* attempt fails, one explicitly marked emergency offer is returned
        rather than silently substituting the template generator or aborting evaluation.
        """
        if n < 1:
            return []
        # Training uses `history_to_prompt(state) + ' '` before the target text.
        # Do NOT change history_to_prompt itself: saved Q encoders use its exact text.
        prompt = history_to_prompt(state) + ' '
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=LM_LENGTH)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prefix_len = inputs['input_ids'].shape[1]
        unique = {}
        empty_count = 0
        duplicate_count = 0
        attempted = 0
        for attempt in range(LM_SAMPLE_ATTEMPTS):
            missing = n - len(unique)
            if missing == 0:
                break
            # More exploratory sampling only when previous attempts have collapsed.
            temperature = (0.85, 1.0, 1.15)[min(attempt, 2)]
            top_p = (0.90, 0.95, 0.98)[min(attempt, 2)]
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs, do_sample=True, top_p=top_p, temperature=temperature,
                    min_new_tokens=2, max_new_tokens=32,
                    num_return_sequences=missing,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            attempted += len(outputs)
            for out in outputs:
                text = self.tokenizer.decode(out[prefix_len:], skip_special_tokens=False)
                # Keep the custom [PRICE] token (skip_special_tokens=True would erase it).
                eos = self.tokenizer.eos_token
                if eos:
                    text = text.split(eos, 1)[0]
                reply = text.strip()
                if reply.lower().startswith('seller:'):
                    reply = reply[len('seller:'):].strip()
                reply = reply.splitlines()[0].strip() if reply else ''
                if not reply or reply.lower().startswith('buyer:'):
                    empty_count += 1
                    continue
                kind = response_type(reply)
                normalized = reply if kind == 'message' else kind
                key = (kind, normalized.strip().casefold())
                if key in unique:
                    duplicate_count += 1
                    continue
                unique[key] = {'kind': kind, 'text': normalized, 'source': 'gpt2'}
        self.last_sampling_stats = {
            'requested': n, 'unique': len(unique), 'attempted': attempted,
            'empty': empty_count, 'duplicates': duplicate_count,
            'fallback': not bool(unique),
        }
        if not unique:
            if not getattr(self, '_fallback_warning_emitted', False):
                warnings.warn(
                    'GPT-2 returned no usable reply after '
                    f'{LM_SAMPLE_ATTEMPTS} bounded sampling attempts ({attempted} outputs). '
                    'Using ONE explicitly flagged emergency offer; evaluation metrics will '
                    'report fallback usage. Check whether the fine-tuned model emits EOS/newlines.',
                    RuntimeWarning, stacklevel=2,
                )
                self._fallback_warning_emitted = True
            return [{'kind': 'offer', 'text': 'offer', 'source': 'gpt2_fallback'}]
        if len(unique) < n and not getattr(self, '_shortfall_warning_emitted', False):
            warnings.warn(
                f'GPT-2 yielded only {len(unique)}/{n} distinct replies after '
                f'{attempted} samples ({empty_count} empty, {duplicate_count} duplicate). '
                'Only unique actions will be scored; no synthetic replies were added.',
                RuntimeWarning, stacklevel=2,
            )
            self._shortfall_warning_emitted = True
        return list(unique.values())


class TemplateGenerator:
    """Explicitly untrained text generator for CI / no-download smoke tests, NOT GPT-2."""
    TEMPLATES = [
        'The item is available and in good condition.',
        'I can offer it for [PRICE].',
        'Would [PRICE] work for you?',
        'Thanks for considering it. I can arrange pickup.',
        'offer',
        'accept',
        'reject',
    ]

    def sample(self, state, n=5):
        # Reproducible with global random seed and independent of model downloads.
        replies = random.sample(self.TEMPLATES, min(n, len(self.TEMPLATES)))
        if n > len(replies):
            replies += random.choices(self.TEMPLATES, k=n-len(replies))
        return [{'kind': response_type(t), 'text': t} for t in replies]
