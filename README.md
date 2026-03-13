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

## Discussion

### Key Findings

**Few-shot prompting significantly improves recall.**
`gpt-4.1` zero-shot achieves a Micro F1 of 0.563, rising to 0.758 with Few-Shot-10 — a 35% relative improvement. The gain is driven primarily by recall (0.518 → 0.866): in-context examples encourage the model to predict toxic labels more actively rather than defaulting to safe, conservative outputs.

**Synthetic few-shot examples are competitive with real ones.**
`gpt-4.1` Few-Shot-10-Synth (0.763) marginally outperforms Few-Shot-10 using real labelled examples (0.758). This suggests that well-constructed synthetic examples can substitute for real annotated data in the prompt — useful when labelled data is scarce or costly to curate.

**`gpt-5.4` has a strong zero-shot baseline.**
Its zero-shot Micro F1 (0.751) is comparable to `gpt-4.1` Few-Shot-10 (0.758), suggesting the model has internalised stronger toxicity-detection priors. Consequently, few-shot prompting yields smaller marginal gains for `gpt-5.4` than for `gpt-4.1`.

**`gpt-5-mini` exhibits a near-zero recall failure.**
Micro recall ranges from 0.018 to 0.116 — the model classifies almost all comments as non-toxic. Notably, Few-Shot-10 (0.067) performs *worse* than Few-Shot-5 (0.202), suggesting that longer contexts confuse the model rather than helping it. This makes `gpt-5-mini` unsuitable for this task without further prompt tuning.

**Fine-tuned models retain a clear performance advantage.**
The best LLM result (`gpt-5.4` Few-Shot-10-Synth: 0.789) still trails `toxic-bert` (0.883) by ~10 percentage points. The gap is most pronounced on rare labels: LLMs produce near-zero F1 on `severe_toxic` and `threat`, whereas `toxic-bert` achieves 0.444 and 0.667 respectively.

### Future Improvements

**Address `gpt-5-mini`'s recall collapse.** The prompt could include an explicit instruction to err on the side of labelling (e.g. *"if in doubt, assign the label"*), or the model's output probabilities could be used to lower the effective classification threshold.

**Explore Chain-of-Thought (CoT) prompting.** Adding a reasoning step before the final label decision may help with ambiguous boundary cases, particularly for `severe_toxic` and `threat` where even fine-tuned models struggle.

**Fine-tune `gpt-4.1` on the Jigsaw training set.** OpenAI's fine-tuning API supports `gpt-4.1`; supervised fine-tuning on toxic comments could close the ~10% gap to specialised BERT models. (Fine-tuning support for the gpt-5 series is currently unconfirmed.)

**Improve few-shot example selection.** The current strategy selects fixed examples covering 10 predefined label combinations. A confusion-matrix-guided approach — selecting examples that target the model's observed error patterns — could yield more targeted improvements.

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
