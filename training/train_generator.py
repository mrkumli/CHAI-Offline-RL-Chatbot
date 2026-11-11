"""Basic supervised GPT-2 fine-tune: loss only on seller response tokens."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.preprocess import read_jsonl, PRICE_TOKEN
from models.generator import history_to_prompt
from config import DATA, CHECKPOINTS, LM_NAME, LM_LENGTH, LM_EPOCHS, LM_BATCH, LM_LR, seed_everything, device


def train_generator(train_file=DATA/'train_transitions.jsonl', val_file=DATA/'validation_transitions.jsonl',
                    output=CHECKPOINTS/'generator', name=LM_NAME, epochs=LM_EPOCHS,
                    batch_size=LM_BATCH, max_train=0):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError('Install dependencies: pip install -r requirements.txt') from exc
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({'additional_special_tokens': [PRICE_TOKEN]})
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device()).train()
    train_rows = read_jsonl(Path(train_file))
    val_rows = read_jsonl(Path(val_file))
    if max_train:
        train_rows = train_rows[:max_train]
    if not train_rows:
        raise ValueError('No seller transitions in training data.')

    def examples(rows):
        result = []
        for row in rows:
            # Use control words as text for offer, accept, reject (paper Section 4.1).
            reply = row['action']['text'] if row['action']['kind'] == 'message' else row['action']['kind']
            context = tokenizer(history_to_prompt(row['state']) + ' ', add_special_tokens=False)['input_ids']
            target = tokenizer(reply + tokenizer.eos_token, add_special_tokens=False)['input_ids']
            if len(target) >= LM_LENGTH:
                continue
            context = context[-(LM_LENGTH-len(target)):]
            ids = context + target
            result.append((ids, [-100] * len(context) + target))
        return result

    dataset = examples(train_rows)
    valid = examples(val_rows)
    if not dataset:
        raise ValueError('No usable seller examples after tokenization.')

    def collate(batch):
        length = max(len(ids) for ids, _ in batch)
        inputs, labels, masks = [], [], []
        for ids, labs in batch:
            pad = length - len(ids)
            inputs.append(ids + [tokenizer.pad_token_id] * pad)
            labels.append(labs + [-100] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return {'input_ids': torch.tensor(inputs), 'labels': torch.tensor(labels),
                'attention_mask': torch.tensor(masks)}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LM_LR)
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            batch = {k: v.to(device()) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f'LM epoch {epoch + 1}: train_loss={sum(losses)/len(losses):.4f}')
    if valid:
        model.eval()
        with torch.no_grad():
            losses = [model(**{k: v.to(device()) for k,v in collate([ex]).items()}).loss.item()
                      for ex in valid[:40]]
        print(f'LM validation loss (first {len(losses)} examples): {sum(losses)/len(losses):.4f}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print('Saved fine-tuned GPT-2 to', output)
    return output
