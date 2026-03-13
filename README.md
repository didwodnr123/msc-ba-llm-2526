# LLM-Based Toxic Comment Classification

A multi-label toxic comment classification system built with prompt engineering. Classifies online comments into six toxicity categories using zero-shot and few-shot prompting via the OpenAI API — no fine-tuning required.

## Results

Evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw dataset.

### Micro F1 by Model × Prompting Mode

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `toxic-bert` *(fine-tuned baseline)* | — | N/A | N/A | N/A | N/A |
| `unbiased-toxic-roberta` *(fine-tuned baseline)* | — | N/A | N/A | N/A | N/A |
| `gpt-5-mini` | — | — | — | — | — |
| `gpt-5.4` | — | — | — | — | — |
| `gpt-4.1` | — | — | — | — | — |

### API Cost (10,000 samples)

| Model | Zero-Shot | Few-Shot-5 | Few-Shot-10 | Few-Shot-5-Synth | Few-Shot-10-Synth |
|-------|-----------|------------|-------------|------------------|-------------------|
| `gpt-5-mini` | — | — | — | — | — |
| `gpt-5.4` | — | — | — | — | — |
| `gpt-4.1` | — | — | — | — | — |

Few-Shot-5/10 use real labelled examples from the training set. Few-Shot-5/10-Synth use LLM-generated synthetic examples (no real data in the prompt).

## Setup

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

# Step by step
python run.py --step preprocess --n_samples 10000
python run.py --step infer --mode zero_shot
python run.py --step infer --mode few_shot_5
python run.py --step infer --mode few_shot_10
python run.py --step infer --mode few_shot_5_synth
python run.py --step infer --mode few_shot_10_synth
python run.py --step evaluate

# Options
python run.py --n_samples 500                                    # adjust sample size
python run.py --models gpt-5-mini gpt-5.4 gpt-4.1            # compare multiple models
```

## Project Structure

```
LLM/
├── src/
│   ├── preprocess.py   # Text cleaning and stratified sampling
│   ├── prompts.py      # Zero-shot / few-shot prompt templates
│   ├── inference.py    # OpenAI API batch inference
│   └── evaluate.py     # Multi-label classification metrics
├── data/
│   └── train.csv       # Jigsaw dataset (~159K comments)
├── results/            # Prediction CSVs and evaluation summary
├── report/
│   └── report.md       # Full project report
├── run.py              # Pipeline entry point
└── requirements.txt
```

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) — Wikipedia talk page comments annotated for six toxicity labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Model & Configuration

### Models

| Model | Notes |
|-------|-------|
| `gpt-4.1` | **Default** — latest GPT-4 series, high capability |
| `gpt-5-mini` | Cost-efficient, fast |
| `gpt-5.4` | Higher accuracy than gpt-5-mini |

```bash
# Single model
python run.py --models gpt-5.4

# Compare all models in one run
python run.py --models gpt-5-mini gpt-5.4 gpt-4.1
```

### Temperature

`temperature=0` is set for models that support it (e.g., `gpt-4.1`) to make outputs **fully deterministic** — the same comment always produces the same prediction, ensuring reproducible evaluation results.

GPT-5 series models (`gpt-5-mini`, `gpt-5.4`) do not support `temperature=0`; the parameter is omitted for these models (default: 1). See `src/inference.py` for details.

## Tech Stack

- **LLM**: GPT-5-mini / GPT-5.4 / GPT-4.1 (OpenAI API)
- **Data**: pandas
- **Evaluation**: scikit-learn
