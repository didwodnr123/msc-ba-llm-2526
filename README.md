# LLM-Based Toxic Comment Classification

A multi-label toxic comment classification system built with prompt engineering. Classifies online comments into six toxicity categories using zero-shot and few-shot prompting via the OpenAI API — no fine-tuning required.

## Results

Evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw dataset.

### Micro F1 by Model × Prompting Mode

**LLM prompt-engineering** (`gpt-4.1` zero-shot = baseline):

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `gpt-4.1` *(baseline)* | 0.563 | 0.718 | 0.758 | 0.750 | 0.763 |
| `gpt-5-mini` | 0.034 | 0.202 | 0.067 | 0.143 | 0.148 |
| `gpt-5.4` | 0.751 | 0.767 | 0.755 | 0.774 | **0.789** |

**Fine-tuned upper bound** (no prompting — direct inference with threshold=0.5):

| Model | Micro F1 |
|-------|----------|
| `toxic-bert` | **0.883** |
| `unbiased-toxic-roberta` | 0.831 |

Micro F1 is used as the primary metric because it aggregates TP/FP/FN across all labels and samples, reflecting overall system performance under label imbalance — where labels like `threat` and `identity_hate` are far rarer than `toxic`. Macro F1 and Exact Match Accuracy are reported as supplementary metrics.

Per-label precision/recall/F1 available in [`results/evaluation_summary.md`](results/evaluation_summary.md)

### Estimated API Cost (10,000 samples)

Estimated from OpenAI list prices (Mar 2026): gpt-4.1 $1.25/$10.00 per 1M tokens in/out, gpt-5-mini $0.25/$2.00, gpt-5.4 $2.50/$15.00.

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `gpt-4.1` | $3.01 | $4.93 | $6.83 | $4.42 | $5.85 |
| `gpt-5-mini` | $0.60 | $0.99 | $1.36 | $0.88 | $1.17 |
| `gpt-5.4` | $5.52 | $9.37 | $13.15 | $8.34 | $11.19 |

Few-Shot-5/10 use real labelled examples from the training set. Few-Shot-5/10-Synth use LLM-generated synthetic examples (no real data in the prompt).

## Experimental Setup

### Dataset & Sampling

- **Source**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) — test set (`data/test.csv` + `data/test_labels.csv`)
- Unlabelled rows (`toxic == -1`) excluded before sampling
- **Stratified sampling**: 5,000 toxic / 5,000 non-toxic (random_state=42)
- "Toxic" defined as having at least one positive label across the 6 categories
- **Text cleaning**: HTML tag removal and whitespace normalization

### Few-Shot Example Selection

- Examples drawn from the **training set only** — no overlap with the evaluation set (leakage-free)
- 10 target label combinations selected to cover diverse toxicity patterns (clean, single-label, and multi-label cases)
- Short comments preferred (≤150 chars) to keep prompt length manageable
- **Synth variants**: hardcoded LLM-generated examples — no real training data in the prompt

### Models

| Model | Notes |
|-------|-------|
| `gpt-4.1` | **Default** — latest GPT-4 series, high capability |
| `gpt-5-mini` | Cost-efficient, fast |
| `gpt-5.4` | Higher accuracy than gpt-5-mini |

### Inference Configuration

| Parameter | Value |
|-----------|-------|
| Max completion tokens | 200 |
| Temperature | 0 for `gpt-4.1` (deterministic); not supported by gpt-5 series (default=1, non-deterministic) |
| Parallel requests | 4 threads (ThreadPoolExecutor) |
| Retry on failure | Exponential backoff, up to 3 attempts |

> **Note**: gpt-5 series results may vary slightly across runs due to non-deterministic sampling.

### Fine-Tuned Model Configuration

- **Library**: [`detoxify`](https://github.com/unitaryai/detoxify)
- **Binary threshold**: 0.5 (probability → 0/1 label)
- **Batch size**: 256

### Evaluation Metrics

- **Micro F1** *(primary)*: aggregates TP/FP/FN across all labels and samples — reflects overall classification performance
- **Macro F1**: per-label F1 averaged unweighted — sensitive to performance on rare labels
- **Exact Match Accuracy**: fraction of samples where all 6 labels are simultaneously correct
- Undefined precision/recall treated as 0 (`zero_division=0`)

## Installation

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure API key**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

## Usage

```bash
# Full pipeline (preprocess → infer → evaluate)
python run.py

# Step by step (leakage-free evaluation on test set)
python run.py --step preprocess --dataset test --n_samples 10000
python run.py --step infer --mode zero_shot                      # default: gpt-4.1
python run.py --step infer --mode few_shot_5
python run.py --step infer --mode few_shot_10
python run.py --step infer --mode few_shot_5_synth
python run.py --step infer --mode few_shot_10_synth
python run.py --step detoxify                                    # fine-tuned BERT baselines
python run.py --step evaluate

# Options
python run.py --n_samples 500                                    # adjust sample size
python run.py --models gpt-4.1 gpt-5-mini gpt-5.4              # compare multiple models
```

## Project Structure

```
LLM/
├── src/
│   ├── preprocess.py          # Text cleaning and stratified sampling
│   ├── prompts.py             # Zero-shot / few-shot prompt templates
│   ├── inference.py           # OpenAI API batch inference
│   ├── detoxify_inference.py  # Fine-tuned BERT/RoBERTa baselines
│   ├── build_few_shot.py      # Few-shot example builder (real + synthetic)
│   └── evaluate.py            # Multi-label classification metrics
├── data/
│   ├── train.csv              # Jigsaw training set (~159K comments)
│   ├── test.csv               # Jigsaw test set (~153K comments)
│   └── test_labels.csv        # Test labels (~63K labelled rows)
├── results/                   # Prediction CSVs and evaluation summary
├── report/
│   └── report.md              # Full project report
├── run.py                     # Pipeline entry point
└── requirements.txt
```

## Vibe Coding Notes

This project was developed using **VSCode + Claude Code** with active use of AI-assisted coding. The following are practical lessons learned during the process.

**Context management**
- Token usage per session must be managed carefully — hitting the context limit forces an unexpected pause
- Use `/compact` periodically to summarise and compress the conversation; without it, the context grows long and responses slow down
- Use `/clear` when switching to a different task or topic to start fresh

**Giving clear instructions**
- The AI performs well only when instructions are explicit — anything left unspecified simply won't be done
- Good documentation and clear upfront instructions are essential; vague prompts produce vague code

**Plan before coding**
- Use `/plan` before asking for code to produce a detailed implementation plan first
- This reduces the chance of the AI making incorrect assumptions and having to rewrite large chunks

**Test small before running full inference**
- If something is wrong after a full 10,000-sample inference run, you have to re-run it — wasting both time and API budget
- Always test on a small sample (e.g., 10–50 inputs) first: verify the API works, the code runs, and the outputs look correct before scaling up
