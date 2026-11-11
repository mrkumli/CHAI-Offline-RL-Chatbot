# CHAI GPT-2 inference/evaluation hotfix

This is a **source-code-only** fix to the existing CHAI prototype. Apply the ZIP in the **parent** of the existing `CHAI-Project` directory. Do **not** delete or replace your trained `checkpoints/` folder.

PowerShell from the folder containing the `CHAI-Project` directory:

```powershell
Expand-Archive -Path .\CHAI-Project-GPT2-Fix-Only.zip -DestinationPath . -Force
cd .\CHAI-Project
python -m pytest -q
python main.py demo --backend gpt2 --policy prop --n-replies 5 --n-prices 5
python main.py evaluate --backend gpt2 --episodes 5 --n-replies 5 --n-prices 5
```

## Changed source files

- `models/generator.py`: prompt separator matches training; bounded sampling retries; remove empty/duplicate replies; explicit emergency fallback warning/source tag.
- `rl/prop.py`: unique response-price actions (no pointless price copies for `accept` or `reject`), preserve source tag.
- `simulation/environment.py`: pass requested candidate counts, track fallback seller turns.
- `evaluation/evaluate.py`: run full policy list and report fallback counts/rate per policy.
- `training/train_q.py`: show accurate distinct candidate count and higher-precision Q scores.
- `main.py`: pass CLI candidate counts into demo/evaluation.
- `config.py`: `LM_SAMPLE_ATTEMPTS` setting.
- `tests/test_pipeline.py`, `tests/test_generation_regression.py`: updated assumptions and new tests.
- `README.md`: explanation of limits and execution.

**Do not interpret results with nonzero fallback counts as clean GPT-2 performance.** The emergency `offer` is a clearly labeled safety mechanism; no replacement language model or Q network is trained.

Local verification in an environment without the user's real GPT-2 checkpoint: 11/11 pytest tests passed, five-policy fixture CLI evaluation completed, and fixture checkpoint hashes were unchanged. Real-checkpoint behavior must be checked on the user's machine.
