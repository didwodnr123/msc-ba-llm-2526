"""
Evaluation metrics module.

- Computes multi-label classification metrics (Accuracy, Precision, Recall, F1)
- Compares zero-shot vs. few-shot performance
- Saves a summary CSV to the results directory
"""

import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_ground_truth(sample_path: str) -> pd.DataFrame:
    return pd.read_csv(sample_path)[['id'] + LABELS]


def load_predictions(pred_path: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    rename = {f'pred_{l}': l for l in LABELS}
    return df[['id'] + [f'pred_{l}' for l in LABELS]].rename(columns=rename)


def compute_metrics(gt: pd.DataFrame, pred: pd.DataFrame) -> tuple[dict, dict]:
    """
    Compute overall and per-label classification metrics.

    Returns:
        overall:   dict of aggregate metrics
        per_label: dict of per-label precision, recall, and F1
    """
    merged = gt.merge(pred, on='id', suffixes=('_true', '_pred'))

    y_true = merged[[f'{l}_true' for l in LABELS]].values
    y_pred = merged[[f'{l}_pred' for l in LABELS]].values

    # Exact match accuracy: proportion of samples where all labels are correct
    exact_match = (y_true == y_pred).all(axis=1).mean()

    overall = {
        'exact_match_accuracy': exact_match,
        'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_recall':    recall_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_f1':        f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall':    recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1':        f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }

    return overall, per_label


def print_report(overall: dict, per_label: dict, mode: str = '') -> None:
    header = f"  Results [{mode}]" if mode else "  Results"
    print(f"\n{'='*52}")
    print(header)
    print(f"{'='*52}")
    print(f"  Exact Match Accuracy : {overall['exact_match_accuracy']:.4f}")
    print(f"  Micro Precision      : {overall['micro_precision']:.4f}")
    print(f"  Micro Recall         : {overall['micro_recall']:.4f}")
    print(f"  Micro F1             : {overall['micro_f1']:.4f}")
    print(f"  Macro F1             : {overall['macro_f1']:.4f}")
    print(f"\n  {'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*52}")
    for label, s in per_label.items():
        print(f"  {label:<20} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f}")


def save_summary(results: dict, out_path: str = 'results/evaluation_summary.csv') -> None:
    """Save a zero-shot vs. few-shot comparison summary to CSV."""
    rows = []
    for mode, (overall, per_label) in results.items():
        row = {'mode': mode}
        row.update(overall)
        for label, s in per_label.items():
            for metric, val in s.items():
                row[f'{label}_{metric}'] = val
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSummary saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--sample',    default='results/sample.csv')
    parser.add_argument('--pred_zero', default='results/predictions_zero_shot.csv')
    parser.add_argument('--pred_few',  default='results/predictions_few_shot.csv')
    args = parser.parse_args()

    gt = load_ground_truth(args.sample)
    results = {}

    for mode, path in [('zero_shot', args.pred_zero), ('few_shot', args.pred_few)]:
        try:
            pred = load_predictions(path)
            overall, per_label = compute_metrics(gt, pred)
            print_report(overall, per_label, mode=mode)
            results[mode] = (overall, per_label)
        except FileNotFoundError:
            print(f"\n[{mode}] Prediction file not found (skipping): {path}")

    if results:
        save_summary(results)
