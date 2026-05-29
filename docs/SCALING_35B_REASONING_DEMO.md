# 35B-Teacher Reasoning Demonstration Plan

This repository cannot honestly pretrain a 35B-equivalent model from scratch on
3 A100s. The credible demonstration path is teacher distillation: use a local
35B Qwen/Ollama model to generate high-quality reasoning traces, then train the
novel plasma architecture on tokenizer-backed packed shards and evaluate the
result as a small distilled reasoning model.

## Local Teacher

The current host has several Ollama teachers available, including:

- `qwen3.6:35b`
- `omnius-qwen36-35b:latest`
- `huihui_ai/Qwen3.6-abliterated:35b`

Use `qwen3.6:35b` as the default teacher unless you are intentionally testing a
variant.

## Pipeline

### 1. Generate synthetic reasoning traces

Start small to validate formatting:

```bash
PYTHONPATH=src python scripts/generate_synthetic_reasoning.py   --model qwen3.6:35b   --seed-prompts prompts/reasoning_seeds.jsonl   --out data/reasoning/qwen36_reasoning_smoke.jsonl   --count 20
```

Scale generation once the smoke sample looks good:

```bash
PYTHONPATH=src python scripts/generate_synthetic_reasoning.py   --model qwen3.6:35b   --seed-prompts prompts/reasoning_seeds.jsonl   --out data/reasoning/qwen36_reasoning.jsonl   --count 50000   --num-predict 768   --temperature 0.7
```

For stronger coverage, add domain-specific seed files for math, code, logic,
scientific explanation, and system design.

### 2. Build tokenizer-backed packed shards

Install optional scale dependencies first:

```bash
pip install -e ".[scale]"
```

Then pack the synthetic corpus. Qwen tokenizers are a reasonable starting point
because the teacher is Qwen-family:

```bash
PYTHONPATH=src python scripts/build_token_shards.py   --input data/reasoning/qwen36_reasoning.jsonl   --tokenizer Qwen/Qwen2.5-1.5B   --out data/packed/qwen36_reasoning_qwen_tok   --text-column text   --validation-every 100
```

The result is `train.bin`, `validation.bin`, and `meta.json` for memory-mapped
training.

### 3. Train the plasma distillation model

Start with the 300M-class config:

```bash
bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml --steps 1
bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml
```

Monitor:

```bash
./start.sh
# Or from the dashboard Command Console: Check 300M convergence.
PYTHONPATH=src python scripts/convergence_check.py --run logs/a100_3gpu_plasma_300m_qwen_distill
```

Generate samples:

```bash
PYTHONPATH=src python scripts/generate.py   --ckpt logs/a100_3gpu_plasma_300m_qwen_distill/ckpt_final.pt   --prompt "Solve step by step: If 3x + 7 = 31, what is x?"   --device cuda   --max-new-tokens 256
```

### 4. Evaluate the demonstration

The demonstration should be judged against a focused target:

- Can the novel architecture learn teacher-style reasoning traces?
- Does it produce coherent, formatted explanations?
- Does it answer held-out synthetic tasks better than a same-size vanilla
  baseline trained on the same packed shards?
- Does long-context stability remain better than vanilla at the same scale?

Do not claim full 35B parity unless an eval suite supports it.

## Scaling Guidance

Use the estimator before committing long runs:

```bash
PYTHONPATH=src python scripts/estimate_scale.py   --params 3e8   --tokens 2e9   --seq-len 2048   --global-batch 3
```

On this 3x A100 box, a 300M-class teacher-distilled demo is the practical target.
A 1B-class run is possible but should be treated as experimental until memory and
throughput are measured with `--steps 1`. A true 35B-from-scratch or full 35B
capability match is outside this hardware envelope.

## Data Quality Rules

- Keep raw teacher JSONL with prompt, response, teacher model, and timestamp.
- Inspect random samples before packing.
- Deduplicate prompts and reject empty/failed responses.
- Keep validation prompts disjoint from training prompts.
- Mix task families deliberately; do not rely on one generic prompt template.
- Track teacher model tag in every record for reproducibility.
