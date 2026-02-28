# LLM-Based Toxic Comment Classification

A multi-label toxic comment classification system built with prompt engineering. Classifies online comments into six toxicity categories using zero-shot and few-shot prompting via the OpenAI API — no fine-tuning required.

## Results

Evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw dataset.

| Metric | Zero-Shot | Few-Shot-5 | Few-Shot-10 |
|--------|-----------|------------|-------------|
| Exact Match Accuracy | 0.548 | 0.578 | **0.581** |
| Micro Precision | **0.713** | 0.699 | 0.694 |
| Micro Recall | 0.605 | 0.760 | **0.770** |
| Micro F1 | 0.654 | 0.728 | **0.730** |
| Macro F1 | 0.516 | 0.597 | **0.599** |
| API Cost (GPT-4o-mini) | $0.25 | $0.42 | $0.58 |

Few-Shot-5 and Few-Shot-10 refer to prompts with 5 and 10 labelled examples respectively.

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
python run.py --step evaluate

# Options
python run.py --n_samples 500   # adjust sample size
python run.py --model gpt-4o    # change model
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

### Model — GPT-4o-mini

`gpt-4o-mini` was chosen as the default model for the following reasons:

- **Cost-efficient**: significantly cheaper than GPT-4o while retaining strong instruction-following ability
- **Fast**: low latency makes it practical for batch inference over hundreds of samples
- **Sufficient capability**: classification tasks with structured output constraints do not require the full capacity of larger models

Alternative models can be used via the `--model` flag:

| Model | Notes |
|-------|-------|
| `gpt-4o-mini` | Default — best cost/performance balance |
| `gpt-4o` | Higher accuracy, higher cost |
| `gpt-4-turbo` | Strong reasoning, higher cost |

```bash
python run.py --model gpt-4o
```

### Temperature — 0

`temperature=0` was set to make outputs **fully deterministic**. Since this task requires the model to output a fixed set of labels in a consistent format, randomness is undesirable — the same comment should always produce the same prediction, ensuring reproducible evaluation results.

Raising the temperature (e.g., `0.5`–`1.0`) would introduce variability in outputs, which could degrade parsing reliability and make results non-reproducible. To change it, edit `src/inference.py:43`.

## Tech Stack

- **LLM**: GPT-4o-mini (OpenAI API)
- **Data**: pandas
- **Evaluation**: scikit-learn
