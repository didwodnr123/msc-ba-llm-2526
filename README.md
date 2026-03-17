# LLM-Based Toxic Comment Classification

A multi-label toxic comment classification system built with prompt engineering. Classifies online comments into six toxicity categories using zero-shot and few-shot prompting via the OpenAI API — no fine-tuning required.

## Results

Evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw dataset.

### Micro F1 by Model × Prompting Mode

**LLM prompt-engineering** (`gpt-4.1` zero-shot = baseline):

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `gpt-4.1` *(baseline)* | 0.575 | 0.724 | 0.712 | 0.728 | 0.727 |
| `gpt-4.1-mini` | 0.739 | **0.759** | 0.731 | 0.745 | 0.736 |
| `gpt-5.4` | 0.717 | 0.753 | 0.710 | 0.732 | 0.741 |

**Fine-tuned upper bound** (no prompting — direct inference with threshold=0.5):

| Model | Micro F1 |
|-------|----------|
| `toxic-bert` | 0.823 |
| `unbiased-toxic-roberta` | **0.834** |

Micro F1 is used as the primary metric because it aggregates TP/FP/FN across all labels and samples, reflecting overall system performance under label imbalance — where labels like `threat` and `identity_hate` are far rarer than `toxic`. Macro F1 and Exact Match Accuracy are reported as supplementary metrics.

Per-label precision/recall/F1 available in [`results/evaluation_summary.md`](results/evaluation_summary.md)

### Estimated API Cost (10,000 samples)

Estimated from OpenAI list prices (Mar 2026): gpt-4.1 $1.25/$10.00 per 1M tokens in/out, gpt-4.1-mini $0.40/$1.60, gpt-5.4 $2.50/$15.00.

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `gpt-4.1` | $3.01 | $4.93 | $6.83 | $4.42 | $5.85 |
| `gpt-4.1-mini` | $0.56 | $1.27 | $1.90 | $1.04 | $1.47 |
| `gpt-5.4` | $5.52 | $9.37 | $13.15 | $8.34 | $11.19 |

Few-Shot-5/10 use real labelled examples from the training set. Few-Shot-5/10-Synth use LLM-generated synthetic examples (no real data in the prompt).

### Estimated Inference Time (10,000 samples, 4 parallel workers)

Measured on a single prompting mode (zero-shot). Fine-tuned models run locally with no API call overhead.

| Model | Per mode | Full run (5 modes) |
|-------|----------|--------------------|
| `gpt-4.1` | ~29 min | ~2.5 hr |
| `gpt-4.1-mini` | ~24 min | ~2 hr |
| `gpt-5.4` | ~46 min | ~4 hr |
| `toxic-bert` | — | ~2 hr (no prompting modes; CPU inference) |
| `unbiased-toxic-roberta` | — | ~2 hr (no prompting modes; CPU inference) |

## Discussion

### Key Findings

**Few-shot prompting significantly improves recall.**
`gpt-4.1` zero-shot achieves a Micro F1 of 0.575, rising to 0.728 with Few-Shot-5-Synth — a 27% relative improvement. The gain is driven primarily by recall (0.522 → 0.772): in-context examples encourage the model to predict toxic labels more actively rather than defaulting to safe, conservative outputs.

**Synthetic few-shot examples are competitive with real ones.**
`gpt-4.1` Few-Shot-5-Synth (0.728) outperforms Few-Shot-5 using real labelled examples (0.724) and Few-Shot-10 (0.712). This suggests that well-constructed synthetic examples can substitute for real annotated data in the prompt — useful when labelled data is scarce or costly to curate.

**`gpt-5.4` has a strong zero-shot baseline.**
Its zero-shot Micro F1 (0.717) is comparable to `gpt-4.1` Few-Shot-5 (0.724), suggesting the model has internalised stronger toxicity-detection priors. Consequently, few-shot prompting yields smaller marginal gains for `gpt-5.4` than for `gpt-4.1`.

**`gpt-4.1-mini` matches or exceeds `gpt-5.4` across all modes.**
`gpt-4.1-mini` achieves the highest LLM result overall (Few-Shot-5: 0.759) and outperforms `gpt-5.4` (0.753) in the same mode. However, this comparison is not fully controlled: `gpt-4.1` and `gpt-4.1-mini` use `temperature=0` (deterministic), whereas `gpt-5.4` does not support `temperature` and defaults to `temperature=1` (non-deterministic). The 0.6-point difference is within the expected run-to-run variance of `gpt-5.4` and should not be interpreted as a reliable performance gap.

**`gpt-5-mini` requires a higher token limit due to internal reasoning.**
Unlike `gpt-4.1` and `gpt-5.4`, `gpt-5-mini` is a reasoning model that uses 64–640 internal reasoning tokens per request before producing output. The initial results (near-zero recall) were caused by `max_completion_tokens=30` being exhausted entirely by reasoning, leaving no tokens for the actual label output. After correcting this to `max_completion_tokens=1000`, the model produces valid predictions.

**Fine-tuned models retain a clear performance advantage.**
The best LLM result (`gpt-4.1-mini` Few-Shot-5: 0.759) still trails `unbiased-toxic-roberta` (0.834) by ~7 percentage points. The gap is most pronounced on rare labels: LLMs produce low F1 on `severe_toxic` and `threat`, whereas `unbiased-toxic-roberta` achieves 0.435 and 0.570 respectively. `unbiased-toxic-roberta` (0.834) outperforms `toxic-bert` (0.823) overall, driven by stronger recall on the `toxic` label (0.959 vs 0.909).

### Future Improvements

**Explore other reasoning models with corrected token limits.** The `gpt-5-mini` failure was caused by `max_completion_tokens=30` being exhausted by internal reasoning tokens. With a corrected budget (e.g. `max_completion_tokens=1000`), reasoning models may produce valid predictions — though the reasoning overhead still makes them slower and more expensive per request.

**Fine-tune `gpt-4.1` on the Jigsaw training set.** OpenAI's fine-tuning API supports `gpt-4.1`; supervised fine-tuning on toxic comments could close the ~10% gap to specialised BERT models. (Fine-tuning support for the gpt-5 series is currently unconfirmed.)

**Improve few-shot example selection.** The current strategy selects fixed examples covering 10 predefined label combinations. A confusion-matrix-guided adaptive approach — selecting examples that target the model's observed error patterns — could yield more targeted improvements.

**Optimise per-label thresholds for fine-tuned models.** `toxic-bert` and `unbiased-toxic-roberta` currently use a fixed threshold of 0.5 for all labels. Tuning thresholds individually on a held-out validation set could improve F1, especially for rare labels with skewed probability distributions.

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

**LLM (prompt-engineering via API)**

| Model | Type | Provider | Notes |
|-------|------|----------|-------|
| `gpt-4.1` | Non-reasoning | OpenAI | **Default** — latest GPT-4 series, high capability |
| `gpt-4.1-mini` | Non-reasoning | OpenAI | Faster, cheaper variant of gpt-4.1 |
| `gpt-5.4` | Non-reasoning | OpenAI | Higher accuracy; best overall LLM result |

> **Note**: Gemini and Grok (Llama via Groq) were tested but excluded — free-tier rate limits made throughput too slow for 10,000-sample evaluation.

**Fine-tuned (local inference, no API key required)**

| Model | Library | Notes |
|-------|---------|-------|
| `toxic-bert` | detoxify | BERT fine-tuned on Jigsaw dataset |
| `unbiased-toxic-roberta` | detoxify | RoBERTa fine-tuned on Jigsaw; best overall |

#### Why non-reasoning models?

LLMs come in two types: **non-reasoning** models (standard, output immediately) and **reasoning** models (spend additional tokens thinking internally before responding — e.g. OpenAI o1, gpt-5-mini, gpt-5-nano).

For this task — outputting a short comma-separated label string — reasoning models are a poor fit:

- Label output is extremely short (~10–18 tokens: `"toxic, insult"`)
- Reasoning models consume 64–1,600 internal tokens *before* any output
- Reasoning overhead accounts for ~99% of total token usage, making them slower and more expensive with no benefit
- `gpt-5-mini` was initially tested and produced near-zero recall across all modes — the `max_completion_tokens=30` budget was entirely consumed by internal reasoning, leaving no tokens for the actual label output (`content = ''`, `finish_reason = 'length'`). Even after correcting to `max_completion_tokens=1000`, each request used 64–704 reasoning tokens for a task that needed only 15 output tokens

Non-reasoning models (gpt-4.1 family) produce output immediately, use `max_completion_tokens=30`, and support `temperature=0` for deterministic results.

### Inference Configuration

| Parameter | Value |
|-----------|-------|
| Max completion tokens | 30 |
| Temperature | 0 (deterministic) |
| Parallel requests | 4 threads (ThreadPoolExecutor) |
| Retry on failure | Exponential backoff, up to 3 attempts |

> **Note**: `gpt-5.4` does not support `temperature=0` (gpt-5 series default is 1, non-deterministic). Results may vary slightly across runs, though variance is negligible at 10,000 samples.

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
python run.py --models gpt-4.1 gpt-4.1-mini gpt-5.4            # compare multiple models
python run.py --workers 8                                        # increase parallel requests
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
├── figures/                   # Generated plots (EDA and results visualisation)
├── eda.ipynb                  # Exploratory data analysis notebook
├── results_viz.ipynb          # Results visualisation notebook
├── run.py                     # Pipeline entry point
└── requirements.txt
```

## Vibe Coding Notes

The full list of structured prompts used during AI-assisted development is in [`vibe_coding_prompts.md`](vibe_coding_prompts.md).


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
