# CHAI offline reinforcement learning — Week 13 prototype

**Status:** An independent PyTorch teaching-oriented prototype, **not** a reproduction of published scores. Implements the CHAI proposal sampling, sampled CQL regularization and KL-regularized Gaussian price BRAC ideas from [Verma et al. (NAACL 2022)](https://aclanthology.org/2022.naacl-main.332/), sections 3–4 and Appendix A.1–A.2. No source code from the CHAI authors is imported or copied.

**Data:** [CraigslistBargain](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) by He et al. The Stanford HF dataset card reports 5,247 train / 597 validation / 838 test dialogues. The canonical Cocoa source JSON may be downloaded using `prepare --download`; if CodaLab is unavailable, download the three source JSONs from its linked CodaLab bundles and place at `data/raw/{train,validation,test}.json`. This project keeps the official splits and does not fabricate missing test dialogue-act labels. The original dataset's license is unspecified on its card; check usage conditions before distributing raw files.

## Installation

Python 3.10+; virtual environment suggested. For NVIDIA GPU, install the correct CUDA build of PyTorch from https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt`. On CPU, all modules run, but GPT-2 fine-tuning is slow. Transformers and GPT-2 download require Internet on the machine/Colab that trains.

```bash
cd CHAI-Project
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Actual dataset + GPT-2 pipeline (intended interim demonstration)

```bash
python main.py prepare --download --limit 200
python main.py train-lm --model gpt2 --epochs 1 --batch-size 2 --max-train 400
python main.py train-q --method prop --backend gpt2 --encoder gpt2 --epochs 2 --max-train 200
python main.py train-q --method cql --backend gpt2 --encoder gpt2 --epochs 2 --max-train 200
python main.py train-q --method brac --backend gpt2 --encoder gpt2 --epochs 2 --max-train 200
python main.py demo --policy prop --backend gpt2 --buyer cooperative
python main.py evaluate --backend gpt2 --episodes 5
```

**Do not use `--limit` for final experiments:** it's a development setting. Run without `--limit` for the complete official splits. Use the same data, generated proposals, encoder and evaluation scenarios across all three methods. Check that the CodaLab source URLs remain accessible. If using GPT-2-medium, change `--model gpt2-medium` and ensure enough GPU memory. Each method saves independent Q checkpoint `checkpoints/q_{method}.pt`.

**Inference vs training:** GPT-2 is trained only in `train-lm`. The Q critic and BRAC Gaussian actor are trained from *logged transitions*, never via environment interaction. The simulator is used **only for evaluation**. The plain LM baseline samples candidates but does not consult Q. The Q policies rank response-price cross-products using a Q-value softmax, as in paper Section 4.2.

## No-internet smoke test (does not need transformers or GPT-2 weights)

```bash
python main.py smoke
python main.py demo --backend template --data-dir data/fixture_processed --policy brac --buyer stingy
pytest -q
```

`smoke` saves an explicitly **synthetic fixture** to `data/fixture_processed` and untrained-language-model test Q checkpoints to `checkpoints/fixture/` (it does not overwrite the real processed dataset). Its output is `results/fixture_smoke.json`; NEVER report it as a performance number for CraigslistBargain or GPT-2. Choose `--data-dir data/fixture_processed` only when explicitly testing the fixture. `--backend template` is *only* a functional test substitute for a learned language model; its Q checkpoints are isolated under `checkpoints/fixture/`.

## Modules / code walkthrough

- `data/preprocess.py`: load source JSON or exported HF rows, replace price mentions with `[PRICE]`, identify seller turns, construct `s, a, r, s', done`; terminal `10 * final/list` reward or `-20` rejection penalty.
- `models/generator.py` and `training/train_generator.py`: GPT-2 response template generation and masked-label fine-tuning. Offers/accepts/rejects become control words, as described by Verma et al.
- `rl/features.py`, `models/q_network.py`: state and action mean-pooled GPT-2 hidden features (when `--encoder gpt2`), normalized prices, one-hot act types, feed-forward Q critic, target soft updates.
- `rl/prop.py`: language-model action proposals and uniformly sampled prices (0.7 to 1.0 of last offer); max over sampled next Q.
- `rl/cql.py`: sampled log-sum-exp penalty minus logged Q; uses same prop target.
- `rl/brac.py`: Gaussian price actor, fitted linear behavior price prior, differentiable Q minus analytic Gaussian KL, Monte Carlo mean target.
- `simulation/buyers.py`, `simulation/environment.py`: three independently parameterized heuristic buyer agents and terminal agreement/rejection.
- `evaluation/evaluate.py`: normalized revenue (failed deals count zero), acceptance, seller reward, turn count, simple output nonemptiness/diversity. **Human-rated coherence is not automated here.**

## Differences from the paper (required in viva)

1. `gpt2` (124M parameters) is chosen instead of the authors' final GPT-2 medium; far less training and fewer dialogues. No comparative claims are warranted until genuine training/evaluation completes.
2. `--encoder hashed` is a deterministic CPU fallback for critic smoke testing, **not** the paper architecture. Use `--encoder gpt2` to compute masked mean hidden states. The frozen, expensive GPT-2 representations are cached in memory by text while training; no end-to-end Q-gradient into the language model.
3. Proposal samples are cached **once** per training transition, and the sampled CQL logsumexp approximates a sum over the immense action space. The price/utterance sample count and some hyperparameters are practical approximations.
4. The BRAC behavior price Gaussian uses a least-squares conditional mean and approximate conditional standard deviation; actor updates only when a logged seller offer exists; prices are clipped. It implements the correct *type* of BRAC objective, but not an exact numerical reproduction. BRAC uses sampled LM act types instead of the paper's uniform type sampling.
5. Original CodaLab data contains incomplete/ambiguous outcomes. This code assigns zero terminal reward when no explicit accept/reject exists; it does **not** invent successful deals. The standard test split lacks dialogue-act metadata, so source event action labels are used if present. Offer/control mapping and masked numeric fields should be inspected before experimental claims.
6. Buyer bots are transparent handcrafted approximations, **not** the paper's bots, nor human subjects. The `nonempty_reply_rate` and `distinct_reply_fraction` are crude automatic diagnostics, *not* human fluency/coherence judgments. Human study, liar/consistency detection, stronger models and full dataset runs are final-demo extensions.
7. The average price on successful sales is an *unnormalized* dollar amount, while normalized revenue uses sale price / list price and zero on failures. Do not compare these values directly to paper Table 1.

## Suggested interim live demo

1. Show one processed JSONL dialogue plus transitions: `data/processed/train_transitions.jsonl` (fixture at `data/fixture_processed/`).
2. Open code for prop, CQL, BRAC and explain Bellman targets / different regularizers.
3. Run `demo` to print the list price, all candidate reply-price Q-values, selected response and full buyer-seller transcript.
4. Show `results/evaluation.json` only if it was generated from **real dataset and trained GPT-2**. Otherwise show `fixture_smoke.json` and clearly label it as a systems integration test.

## Reproducibility

`seed_everything(42)` controls Python, NumPy and Torch sampling; GPU kernels and HF generation may still vary across machines. Maintain a record of library versions, command lines, training data limits and checkpoint SHA256 hashes. No official dataset or pretrained model files are bundled.

## GPT-2 evaluation crash fix (November 2026)

The original generator was fine-tuned on a prompt ending in `Seller: ` (with a space) but sampled from `Seller:` (without a space). Sampling could emit only EOS or a blank line and would then raise `RuntimeError`. Inference now matches the training prompt **without changing** `history_to_prompt`, which is used by the saved Q encoders. It tries at most three sampled batches (higher temperature/top-p on later attempts), removes blank/duplicated responses, and reports the shortfall.

If **all** retries still yield nothing, a *single, explicitly labeled emergency `offer` action* is substituted with `source=gpt2_fallback`, and a `RuntimeWarning` is printed. This is **not** a GPT-2-generated response and is **not** the template generator. `results/evaluation.json` reports `fallback_seller_turns` and `fallback_seller_turn_rate` for **each** policy. Results using fallbacks must not be presented as clean GPT-2 results.

Duplicate `accept`/`reject` actions no longer get multiplied by `--n-prices`. Q-scores for identical `(state, action)` inputs were necessarily identical; the old 25-entry list did not show 25 genuine alternative actions. The new demo prints **unique** candidate count and Q-values to six decimal places. If your checkpoint genuinely produces only `accept` after resampling, there is only **one** unique action to score; the system doesn't fabricate variety. You can investigate response diversity separately before relying on evaluation metrics.

The CLI now actually forwards `--n-replies` and `--n-prices` to both demo and evaluation. No existing `.pt` critic checkpoints or GPT-2 model files need to be retrained or modified for these changes.

**After applying the bugfix to the existing folder (do not overwrite `checkpoints/`):**

```bash
python -m pytest -q
python main.py demo --backend gpt2 --policy prop --n-replies 5 --n-prices 5
python main.py evaluate --backend gpt2 --episodes 5 --n-replies 5 --n-prices 5
```

The repository tests mock generation and use lightweight fixture models. They do not confirm how *your particular saved GPT-2 weights* behave; examine the console warnings and JSON fallback counts from an actual GPT-2 evaluation.
