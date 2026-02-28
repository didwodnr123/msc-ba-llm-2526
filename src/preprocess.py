"""
Data preprocessing and sample extraction module.

- Loads and cleans text from train.csv
- Performs stratified sampling to maintain toxic/non-toxic ratio
"""

import re
import pandas as pd

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def clean_text(text: str) -> str:
    """Remove HTML tags and normalise excessive whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML tags
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text


def load_and_sample(
    data_path: str = 'data/train.csv',
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load dataset and extract a balanced evaluation subset.

    Samples toxic and non-toxic comments at a 1:1 ratio to avoid
    class imbalance skewing the evaluation metrics.
    """
    df = pd.read_csv(data_path)
    df['comment_text'] = df['comment_text'].astype(str).apply(clean_text)

    # A comment is considered toxic if it has at least one positive label
    df['is_toxic'] = (df[LABELS].sum(axis=1) > 0).astype(int)

    toxic_df = df[df['is_toxic'] == 1]
    clean_df  = df[df['is_toxic'] == 0]

    n_toxic = min(n_samples // 2, len(toxic_df))
    n_clean = n_samples - n_toxic

    sampled = pd.concat([
        toxic_df.sample(n=n_toxic, random_state=random_state),
        clean_df.sample(n=n_clean,  random_state=random_state),
    ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sampling complete: {len(sampled)} comments (toxic: {n_toxic} / non-toxic: {n_clean})")
    print("Label distribution:")
    print(sampled[LABELS].sum().to_string())

    return sampled[['id', 'comment_text'] + LABELS]


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    df = load_and_sample(n_samples=500)
    df.to_csv('results/sample.csv', index=False)
    print("\nSaved: results/sample.csv")
