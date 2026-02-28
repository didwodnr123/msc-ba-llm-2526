# LLM-Based Toxic Comment Classification: Project Report

---

## 1. Introduction

This report presents the methodology, experimental results, and analysis for an LLM-based toxic comment classification system built using prompt engineering. The system classifies online comments into six toxicity categories without any model fine-tuning, relying solely on zero-shot and few-shot prompting via the OpenAI API.

The primary objectives of this project are:

- To evaluate the effectiveness of zero-shot prompting for multi-label toxicity classification
- To assess whether few-shot prompting yields measurable improvements over zero-shot
- To report classification performance using standard multi-label metrics

---

## 2. Dataset

The **Jigsaw Toxic Comment Classification** dataset (sourced from Kaggle) was used for evaluation. It consists of Wikipedia talk page comments annotated for six toxicity categories.

| Property | Detail |
|----------|--------|
| Source | Kaggle / HuggingFace |
| Total size | ~159,000 annotated comments |
| Labels | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |
| Task type | Multi-label classification |

**Evaluation subset:** 500 comments were sampled using stratified sampling (250 toxic, 250 non-toxic) to ensure a balanced evaluation without class imbalance bias.

**Label distribution in the evaluation subset:**

| Label | Count |
|-------|-------|
| toxic | 242 |
| obscene | 131 |
| insult | 127 |
| identity_hate | 24 |
| severe_toxic | 21 |
| threat | 9 |

---

## 3. Methodology

### 3.1 Data Preprocessing

Raw comment text was cleaned prior to inference:
- HTML tags were removed using regular expressions
- Consecutive whitespace characters were collapsed into a single space

No further normalisation (e.g., lowercasing, stopword removal) was applied, as LLMs operate effectively on natural text.

### 3.2 Prompt Design

Two prompting strategies were evaluated:

#### Zero-Shot Prompt

The model was provided with the list of labels and instructed to output only the applicable ones, with no examples.

```
System:
You are a content moderation assistant. Classify the given comment
using these toxicity labels: toxic, severe_toxic, obscene, threat,
insult, identity_hate. Output ONLY the applicable labels,
comma-separated. If none apply, output exactly: none.
Do not explain or add any other text.

User:
Comment: "{comment_text}"
```

#### Few-Shot Prompt

Five representative examples (one per major category, plus one clean example) were prepended to the user message to guide the model's output format and decision boundaries.

```
Comment: "You are a complete moron and nobody likes you."
Labels: toxic, insult

Comment: "I know where you live. You better watch your back."
Labels: toxic, threat

Comment: "This is absolute garbage!! You disgusting piece of filth!!"
Labels: toxic, severe_toxic, obscene, insult

Comment: "Go back to your country, you don't belong here."
Labels: toxic, identity_hate

Comment: "Thank you for the clarification. I appreciate your help."
Labels: none

Comment: "{comment_text}"
Labels:
```

### 3.3 Model and Inference

| Setting | Value |
|---------|-------|
| Model | GPT-4o-mini |
| Temperature | 0 (deterministic output) |
| Max tokens | 50 |
| Inference | Parallelised via `ThreadPoolExecutor` (4 workers) |
| Fine-tuning | None — inference only |

LLM responses were parsed by splitting on commas and matching tokens against the known label set. Unrecognised tokens were silently ignored; an empty or `none` response was treated as all-negative.

---

## 4. Results

### 4.1 Overall Metrics

| Metric | Zero-Shot | Few-Shot | Δ |
|--------|-----------|----------|---|
| Exact Match Accuracy | 0.544 | **0.594** | +0.050 |
| Micro Precision | **0.704** | 0.699 | −0.005 |
| Micro Recall | 0.626 | **0.791** | +0.165 |
| Micro F1 | 0.663 | **0.742** | +0.079 |
| Macro F1 | 0.493 | **0.585** | +0.092 |

### 4.2 Per-Label Results

#### Zero-Shot

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| toxic | **0.962** | 0.628 | 0.760 |
| severe_toxic | 0.000 | 0.000 | 0.000 |
| obscene | **0.935** | 0.443 | 0.601 |
| threat | 0.385 | 0.556 | 0.455 |
| insult | 0.579 | **0.890** | 0.702 |
| identity_hate | 0.306 | 0.792 | 0.442 |

#### Few-Shot

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| toxic | **0.947** | **0.880** | **0.912** |
| severe_toxic | 0.209 | 0.429 | **0.281** |
| obscene | **0.883** | **0.634** | **0.738** |
| threat | 0.357 | 0.556 | 0.435 |
| insult | 0.578 | **0.850** | **0.688** |
| identity_hate | 0.313 | **0.833** | **0.455** |

---

## 5. Analysis

### 5.1 Few-Shot vs. Zero-Shot

Few-shot prompting delivers consistent improvements across almost all metrics. The most significant gain is in **Micro Recall (+0.165)**, indicating that the model misses fewer toxic comments when guided by examples. Micro F1 improves by 0.079 and Macro F1 by 0.092, demonstrating that few-shot prompting benefits both aggregate and per-class performance.

The only trade-off is a marginal drop in Micro Precision (−0.005), suggesting that a small number of additional false positives are introduced — an acceptable cost given the substantial recall improvement.

### 5.2 Label-Level Observations

**`toxic`** is the best-performing label in both modes (F1: 0.760 → 0.912). This is expected, as it is the broadest and most frequently represented category, and its definition is most consistent with general LLM training data.

**`severe_toxic`** is the most challenging label. Zero-shot produces an F1 of 0.000, failing to detect any instances. Few-shot raises this to 0.281 — a significant improvement, but still limited by the small number of positive examples in the evaluation subset (21 out of 500).

**`threat`** achieves moderate F1 scores in both modes (0.455 and 0.435). The very low sample count (9 positives) makes reliable measurement difficult and likely inflates variance.

**`obscene`** shows high precision (0.935 / 0.883) in both modes but lower recall, suggesting the model is conservative — it labels comments as obscene only when highly confident.

**`insult`** and **`identity_hate`** both show high recall, particularly in few-shot mode, indicating the model is sensitive to these categories when examples are provided.

### 5.3 Error Analysis

Common failure patterns observed in the predictions include:

- **False negatives on `severe_toxic`:** The model under-detects extreme toxicity in zero-shot mode, likely because boundary cases between `toxic` and `severe_toxic` are ambiguous without examples.
- **False positives on `identity_hate`:** High recall but moderate precision suggests the model occasionally flags culturally specific language as identity-based hate.
- **Conservative `obscene` detection:** The model appears to require explicit profanity rather than implied vulgarity to assign the `obscene` label.

---

## 6. Conclusion

This project demonstrates that LLMs can perform competitive multi-label toxic comment classification through prompt engineering alone, without fine-tuning.

Key findings:

1. **Few-shot prompting significantly outperforms zero-shot**, with Micro F1 improving from 0.663 to 0.742.
2. **High-frequency labels** (`toxic`, `insult`, `obscene`) achieve strong performance; **low-frequency labels** (`severe_toxic`, `threat`) remain difficult due to limited evaluation samples.
3. **GPT-4o-mini** provides an effective balance of cost and performance for this task.

**Future improvements:**

- Increase evaluation sample size, particularly for under-represented labels (`threat`, `severe_toxic`)
- Explore chain-of-thought prompting to improve boundary-case classification
- Investigate label-specific few-shot examples to further improve per-label recall

---

## 7. Deliverables

| Item | Location |
|------|----------|
| Source code & prompt templates | `src/` |
| Evaluation scripts | `src/evaluate.py` |
| Prediction results | `results/predictions_*.csv` |
| Evaluation summary | `results/evaluation_summary.csv` |
| This report | `report/report.md` |

---

*Model: GPT-4o-mini · Evaluation set: 500 comments · Framework: OpenAI Python SDK*
