# LLM-Based Toxic Comment Classification

A multi-label toxic comment classification system built with prompt engineering. Classifies online comments into six toxicity categories using zero-shot and few-shot prompting via the OpenAI API — no fine-tuning required.

## Results

| Metric | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| Exact Match Accuracy | 0.544 | **0.594** |
| Micro F1 | 0.663 | **0.742** |
| Macro F1 | 0.493 | **0.585** |

Evaluated on 500 sampled comments (250 toxic / 250 non-toxic) from the Jigsaw dataset.

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
python run.py --step preprocess --n_samples 500
python run.py --step infer --mode zero_shot
python run.py --step infer --mode few_shot
python run.py --step evaluate

# Options
python run.py --n_samples 200   # adjust sample size
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

## Tech Stack

- **LLM**: GPT-4o-mini (OpenAI API)
- **Data**: pandas
- **Evaluation**: scikit-learn
