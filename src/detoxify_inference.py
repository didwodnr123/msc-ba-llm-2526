"""
Detoxify batch inference module.

Runs Jigsaw-fine-tuned BERT/RoBERTa models locally via the detoxify library.
No API key required.
"""

import pandas as pd
from detoxify import Detoxify

# Maps detoxify output keys → Jigsaw label names
LABEL_MAP = {
    'toxicity':        'toxic',
    'severe_toxicity': 'severe_toxic',
    'obscene':         'obscene',
    'threat':          'threat',
    'insult':          'insult',
    'identity_attack': 'identity_hate',
}

DETOXIFY_MODEL_NAMES = {
    'toxic-bert':             'original',
    'unbiased-toxic-roberta': 'unbiased',
}


def run_detoxify(
    df: pd.DataFrame,
    model_name: str = 'toxic-bert',
    threshold: float = 0.5,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Run a detoxify model on a DataFrame of comments.

    Args:
        df:         DataFrame with 'id' and 'comment_text' columns
        model_name: 'toxic-bert' or 'unbiased-toxic-roberta'
        threshold:  Probability threshold for converting scores to binary labels
        batch_size: Number of comments per forward pass

    Returns:
        DataFrame with columns: id, pred_toxic, pred_severe_toxic, ...
    """
    assert model_name in DETOXIFY_MODEL_NAMES, \
        f"model_name must be one of {list(DETOXIFY_MODEL_NAMES)}, got: {model_name}"

    model = Detoxify(DETOXIFY_MODEL_NAMES[model_name])
    texts = df['comment_text'].tolist()
    total = len(texts)

    all_scores: dict[str, list] = {k: [] for k in LABEL_MAP}

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        scores = model.predict(batch)
        for key in LABEL_MAP:
            vals = scores[key]
            all_scores[key].extend(vals.tolist() if hasattr(vals, 'tolist') else list(vals))

        done = min(start + batch_size, total)
        if done % 500 == 0 or done == total:
            print(f"  Progress: {done}/{total}")

    records = []
    for i, row_id in enumerate(df['id']):
        record = {'id': row_id}
        for detox_col, jigsaw_col in LABEL_MAP.items():
            record[f'pred_{jigsaw_col}'] = int(all_scores[detox_col][i] >= threshold)
        records.append(record)

    return pd.DataFrame(records)
