"""
Extract real few-shot examples from train.csv.
Saves results/few_shot_examples.json for use in prompts.py.

Each example is selected by exact label combination match, excluding
any IDs present in the evaluation sample to prevent data leakage.
"""

import json
import pandas as pd

from src.preprocess import clean_text, LABELS

# Ordered list of target label combinations.
# Indices 0-4  → 5-shot set.
# Indices 0-9  → 10-shot set.
TARGET_COMBINATIONS = [
    [],                                                      # none (clean)
    ['toxic'],                                               # toxic only
    ['toxic', 'insult'],                                     # common multi-label
    ['toxic', 'obscene', 'insult'],                          # most frequent multi-label
    ['toxic', 'identity_hate'],                              # identity-based hate
    ['toxic', 'threat'],                                     # threat coverage
    ['toxic', 'severe_toxic', 'obscene', 'insult'],          # heavily offensive
    ['obscene'],                                             # obscene without toxic
    ['insult'],                                              # insult without toxic
    ['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate'],  # maximum overlap
]


def _labels_to_str(row: pd.Series) -> str:
    active = [l for l in LABELS if row[l] == 1]
    return ', '.join(active) if active else 'none'


def build_few_shot_examples(
    train_path: str = 'data/train.csv',
    exclude_ids=None,
    max_len: int = 150,
    random_state: int = 42,
) -> list[dict]:
    """
    Sample one real example per target label combination.

    Args:
        train_path:   Path to the full training CSV.
        exclude_ids:  Iterable of IDs to exclude (i.e. the evaluation sample).
        max_len:      Maximum comment length (chars) — prefer short comments.
        random_state: Seed for reproducibility.

    Returns:
        List of dicts with keys 'comment' and 'labels'.
    """
    df = pd.read_csv(train_path)
    df['comment_text'] = df['comment_text'].astype(str).apply(clean_text)

    if exclude_ids is not None:
        df = df[~df['id'].isin(set(exclude_ids))]

    examples = []
    for combo in TARGET_COMBINATIONS:
        # Build exact-match mask
        mask = pd.Series(True, index=df.index)
        for label in LABELS:
            expected = 1 if label in combo else 0
            mask &= (df[label] == expected)

        pool = df[mask]
        short = pool[pool['comment_text'].str.len() <= max_len]
        chosen_pool = short if len(short) > 0 else pool

        if len(chosen_pool) == 0:
            print(f"  Warning: no examples found for combination {combo or ['none']}, skipping.")
            continue

        row = chosen_pool.sample(1, random_state=random_state).iloc[0]
        examples.append({
            'comment': row['comment_text'],
            'labels':  _labels_to_str(row),
        })

    return examples


def save_few_shot_examples(
    out_path: str = 'results/few_shot_examples.json',
    **kwargs,
) -> list[dict]:
    """Build and save few-shot examples to a JSON file."""
    examples = build_few_shot_examples(**kwargs)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(examples)} few-shot examples → {out_path}")
    return examples
