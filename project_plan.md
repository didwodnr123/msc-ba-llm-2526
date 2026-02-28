# LLM-Based Toxic Comment Classification — Project Plan

---

## 1. Project Overview

This project aims to build an automated toxic comment classification system using LLMs and prompt engineering. The system performs multi-label classification using inference only — no fine-tuning — and evaluates performance using objective classification metrics.

---

## 2. Dataset

| Property | Detail |
|----------|--------|
| Name | Jigsaw Toxic Comment Classification Dataset |
| Source | Kaggle / HuggingFace |
| Size | ~159,000 Wikipedia comments (local: `data/train.csv`) |
| Task type | Multi-label classification |
| Labels | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |

> Each comment may belong to zero or more of the six categories simultaneously.

---

## 3. Technical Requirements

### 3-1. Data Preprocessing
- Parse comment text and labels from the dataset
- Clean text (remove HTML tags, special characters where necessary)
- Convert to a format suitable for prompt input and evaluation
- Extract an evaluation subset via stratified sampling (mindful of API costs)

### 3-2. Prompt Engineering
- **Zero-shot prompt**: classification using instructions only, without examples
- **Few-shot prompt**: include representative examples for each label
- Output constraint: model must return class labels only

**Zero-shot example**
```
Read the following comment and list all applicable toxicity labels.
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
If none apply, output: none
Output labels only, comma-separated.

Comment: {comment_text}
```

**Few-shot example**
```
Comment: "You are an idiot." → insult, toxic
Comment: "I'll find you." → threat, toxic
Comment: "Great post!" → none

Comment: {comment_text}
```

### 3-3. Model Execution
- Use the OpenAI API (GPT-4o-mini) with the university-provided API key
- Inference only — no fine-tuning
- Parallelise API calls for efficient batch processing
- Response parsing: extract labels from output and convert to binary vector
- Exception handling: implement fallback logic for unexpected outputs

---

## 4. Evaluation Criteria

| Category | Details | Weight |
|----------|---------|--------|
| Classification performance | Accuracy, Precision, Recall, F1 (multi-label) | 40% |
| Prompt effectiveness | Zero-shot vs. few-shot performance comparison | 20% |
| Technical execution | Batch processing, response parsing robustness | 20% |
| Documentation & analysis | Methodology, error case analysis, prompt strategy | 20% |

---

## 5. Implementation Schedule

| Phase | Tasks |
|-------|-------|
| Phase 1 | Environment setup, data loading, EDA, sample extraction |
| Phase 2 | Zero-shot prompt design and inference pipeline implementation |
| Phase 3 | Few-shot prompt design and comparative experiments |
| Phase 4 | Metric computation, error case analysis, results consolidation |
| Phase 5 | Report and presentation preparation, GitHub clean-up |

---

## 6. Project Structure

```
LLM/
├── data/
│   └── train.csv               # Jigsaw dataset (local)
├── src/
│   ├── preprocess.py           # Text cleaning and sample extraction
│   ├── prompts.py              # Zero-shot / few-shot templates
│   ├── inference.py            # LLM API calls and batch processing
│   └── evaluate.py             # Classification metric computation
├── results/                    # Prediction CSVs and evaluation reports
├── report/                     # Final report and presentation materials
├── requirements.txt
└── README.md
```

---

## 7. Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.10+ |
| LLM API | OpenAI GPT-4o-mini (university-provided API key) |
| Data processing | `pandas` |
| Evaluation | `scikit-learn` (`classification_report`, `f1_score`) |
| API calls | `openai` |

---

## 8. Deliverables

- **GitHub repository**: source code, prompt templates, evaluation scripts, prediction results
- **Report / presentation**: methodology, experimental results, error case analysis, prompt strategy
