# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-based multi-label toxic comment classification using prompt engineering (zero-shot and few-shot). No fine-tuning — inference only via the OpenAI API.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (preprocess → infer → evaluate)
python run.py

# Step-by-step
python run.py --step preprocess --n_samples 500
python run.py --step infer --mode zero_shot
python run.py --step infer --mode few_shot
python run.py --step evaluate

# Adjust sample size or model
python run.py --n_samples 200
python run.py --model gpt-4o
```

## Environment

Create a `.env` file (see `.env.example`):
```
OPENAI_API_KEY=your_key_here
```

## Architecture

```
run.py                  # Pipeline orchestrator — entry point
src/
  preprocess.py         # Load train.csv → clean text → stratified sampling
  prompts.py            # Zero-shot / few-shot templates + parse_response()
  inference.py          # OpenAI API batch calls (ThreadPoolExecutor, retry logic)
  evaluate.py           # Multi-label metrics + zero vs. few-shot comparison
data/
  train.csv             # Jigsaw dataset (~159K comments, 6 binary labels)
results/
  sample.csv            # Evaluation subset (500 comments, 1:1 toxic/non-toxic)
  predictions_zero_shot.csv
  predictions_few_shot.csv
  evaluation_summary.csv
report/
  report.md             # Final project report (British English)
```

### Data Flow

`data/train.csv` → `preprocess.load_and_sample()` → `results/sample.csv`
→ `inference.run_inference(mode='zero_shot')` → `results/predictions_zero_shot.csv`
→ `inference.run_inference(mode='few_shot')`  → `results/predictions_few_shot.csv`
→ `evaluate.compute_metrics()` → `results/evaluation_summary.csv`

### Key Design Decisions

- **Labels**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (multi-label)
- **Sampling**: 1:1 toxic/non-toxic stratified sampling to control class imbalance
- **Output format**: Labels only, comma-separated (`none` if clean) — parsed by `parse_response()`
- **Parallelism**: `ThreadPoolExecutor(workers=4)` — no rate-limit delay needed for OpenAI
- **Evaluation**: Exact Match Accuracy, Micro/Macro F1, per-label Precision/Recall/F1
