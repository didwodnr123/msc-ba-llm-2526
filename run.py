"""
Full pipeline entry point.

Usage:
    python run.py                                                   # full run (all models, all modes)
    python run.py --step preprocess --n_samples 10000
    python run.py --step infer --mode zero_shot
    python run.py --step infer --mode few_shot_5
    python run.py --step infer --mode few_shot_10
    python run.py --step detoxify                                   # run fine-tuned BERT/RoBERTa baselines
    python run.py --step evaluate
    python run.py --n_samples 500                                   # adjust sample size
    python run.py --models gpt-4o-mini                              # single model
    python run.py --models gpt-4o-mini gpt-4o o1-mini o3-mini      # multiple models
"""

import glob
import os
import argparse
import pandas as pd

from src.preprocess import load_and_sample, load_test_set
from src.build_few_shot import save_few_shot_examples
from src.inference import run_inference
from src.detoxify_inference import run_detoxify, DETOXIFY_MODEL_NAMES
from src.evaluate import load_ground_truth, load_predictions, compute_metrics, print_report, save_summary


def main():
    parser = argparse.ArgumentParser(description='Toxic Comment Classification Pipeline')
    parser.add_argument('--step',      choices=['preprocess', 'infer', 'detoxify', 'evaluate', 'all'], default='all')
    parser.add_argument('--mode',      choices=['zero_shot', 'few_shot_5', 'few_shot_10', 'few_shot_5_synth', 'few_shot_10_synth', 'all'], default='all')
    parser.add_argument('--n_samples', type=int, default=10000,      help='Number of samples to evaluate')
    parser.add_argument('--models',    nargs='+', default=['gpt-4o-mini'], help='One or more OpenAI model names')
    parser.add_argument('--workers',   type=int, default=4,          help='Number of concurrent API requests')
    parser.add_argument('--dataset',   choices=['train', 'test'], default='train',
                        help='train = sample from train.csv (default); test = use test.csv + test_labels.csv')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    sample_path = 'results/sample.csv'

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    if args.step in ('preprocess', 'all'):
        if args.dataset == 'test':
            print(f"\n[1/3] Preprocessing — loading test set (n_samples={args.n_samples if args.n_samples != 10000 else 'all'})...")
            df = load_test_set(n_samples=args.n_samples)
            save_few_shot_examples()  # test IDs are not in train.csv, so no exclusion needed
        else:
            print(f"\n[1/3] Preprocessing — extracting {args.n_samples} samples...")
            df = load_and_sample(n_samples=args.n_samples)
            save_few_shot_examples(exclude_ids=df['id'].tolist())
        df.to_csv(sample_path, index=False)
        print(f"      Saved: {sample_path}")

    # ── Step 2: Inference ─────────────────────────────────────────────────────
    modes = ['zero_shot', 'few_shot_5', 'few_shot_10', 'few_shot_5_synth', 'few_shot_10_synth'] if args.mode == 'all' else [args.mode]

    if args.step in ('infer', 'all'):
        df = pd.read_csv(sample_path)
        for model in args.models:
            model_slug = model.replace('/', '-')
            for mode in modes:
                print(f"\n[2/3] Inference: {mode} | model={model} | {len(df)} comments")
                preds = run_inference(df, mode=mode, model=model, workers=args.workers)
                out = f'results/predictions_{model_slug}_{mode}.csv'
                preds.to_csv(out, index=False)
                print(f"      Saved: {out}")

    # ── Step 2b: Detoxify Baselines ───────────────────────────────────────────
    if args.step in ('detoxify', 'all'):
        df = pd.read_csv(sample_path)
        for model_name in DETOXIFY_MODEL_NAMES:
            print(f"\n[detoxify] model={model_name} | {len(df)} comments")
            preds = run_detoxify(df, model_name=model_name)
            out = f'results/predictions_{model_name}_detoxify.csv'
            preds.to_csv(out, index=False)
            print(f"      Saved: {out}")

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    if args.step in ('evaluate', 'all'):
        print("\n[3/3] Evaluation...")
        gt = load_ground_truth(sample_path)
        results = {}

        # Auto-discover all prediction files: predictions_{model}_{mode}.csv
        # Model slugs use hyphens (e.g. gpt-4o-mini); modes use underscores (zero_shot, few_shot_5).
        # Splitting on the first underscore reliably separates them.
        valid_modes = {'zero_shot', 'few_shot_5', 'few_shot_10', 'few_shot_5_synth', 'few_shot_10_synth', 'detoxify'}
        pred_files = sorted(glob.glob('results/predictions_*.csv'))
        for path in pred_files:
            filename = os.path.basename(path)           # predictions_gpt-4o-mini_zero_shot.csv
            stem = filename[len('predictions_'):-len('.csv')]   # gpt-4o-mini_zero_shot
            parts = stem.split('_', 1)                  # ['gpt-4o-mini', 'zero_shot']
            if len(parts) != 2 or parts[1] not in valid_modes:
                print(f"  Skipping unrecognised file: {filename}")
                continue
            model_slug, mode = parts
            key = f'{model_slug}/{mode}'
            try:
                pred = load_predictions(path)
                overall, per_label = compute_metrics(gt, pred)
                print_report(overall, per_label, mode=key)
                results[key] = (overall, per_label)
            except Exception as e:
                print(f"  [{key}] Skipping {path}: {e}")

        if results:
            save_summary(results)

    print("\nDone.")


if __name__ == '__main__':
    main()
