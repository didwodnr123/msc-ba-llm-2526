"""
Full pipeline entry point.

Usage:
    python run.py                                  # full run (500 samples, both modes)
    python run.py --step preprocess --n_samples 500
    python run.py --step infer --mode zero_shot
    python run.py --step infer --mode few_shot
    python run.py --step evaluate
    python run.py --n_samples 200                  # adjust sample size
    python run.py --model gpt-4o-mini              # change model
"""

import os
import argparse
import pandas as pd

from src.preprocess import load_and_sample
from src.inference import run_inference
from src.evaluate import load_ground_truth, load_predictions, compute_metrics, print_report, save_summary


def main():
    parser = argparse.ArgumentParser(description='Toxic Comment Classification Pipeline')
    parser.add_argument('--step',      choices=['preprocess', 'infer', 'evaluate', 'all'], default='all')
    parser.add_argument('--mode',      choices=['zero_shot', 'few_shot', 'both'], default='both')
    parser.add_argument('--n_samples', type=int, default=500,        help='Number of samples to evaluate')
    parser.add_argument('--model',     default='gpt-4o-mini',        help='OpenAI model name')
    parser.add_argument('--workers',   type=int, default=4,          help='Number of concurrent API requests')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    sample_path = 'results/sample.csv'

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    if args.step in ('preprocess', 'all'):
        print(f"\n[1/3] Preprocessing — extracting {args.n_samples} samples...")
        df = load_and_sample(n_samples=args.n_samples)
        df.to_csv(sample_path, index=False)
        print(f"      Saved: {sample_path}")

    # ── Step 2: Inference ─────────────────────────────────────────────────────
    modes = ['zero_shot', 'few_shot'] if args.mode == 'both' else [args.mode]

    if args.step in ('infer', 'all'):
        df = pd.read_csv(sample_path)
        for mode in modes:
            print(f"\n[2/3] Inference: {mode} | model={args.model} | {len(df)} comments")
            preds = run_inference(df, mode=mode, model=args.model, workers=args.workers)
            out = f'results/predictions_{mode}.csv'
            preds.to_csv(out, index=False)
            print(f"      Saved: {out}")

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    if args.step in ('evaluate', 'all'):
        print("\n[3/3] Evaluation...")
        gt = load_ground_truth(sample_path)
        results = {}

        for mode in modes:
            path = f'results/predictions_{mode}.csv'
            try:
                pred = load_predictions(path)
                overall, per_label = compute_metrics(gt, pred)
                print_report(overall, per_label, mode=mode)
                results[mode] = (overall, per_label)
            except FileNotFoundError:
                print(f"  [{mode}] Prediction file not found, skipping: {path}")

        if results:
            save_summary(results)

    print("\nDone.")


if __name__ == '__main__':
    main()
