"""
LLM batch inference module.

- Calls the OpenAI API with exponential back-off retry logic
- Supports zero-shot and few-shot prompting modes
- Parallelises requests using ThreadPoolExecutor
"""

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.prompts import (
    ZERO_SHOT_SYSTEM, zero_shot_user,
    FEW_SHOT_SYSTEM, few_shot_5_user, few_shot_10_user,
    parse_response,
)

load_dotenv()
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

from genai_pricing import openai_prompt_cost


def call_llm(
    system: str,
    user: str,
    model: str = 'gpt-4o-mini',
    max_retries: int = 3,
) -> tuple[str, float]:
    """Make a single LLM API call with exponential back-off on failure.

    Returns:
        (response_text, cost_usd)
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user',   'content': user},
                ],
                temperature=0,
                max_tokens=50,
            )
            text = resp.choices[0].message.content or ''
            cost = openai_prompt_cost(model, user, text, resp).get('total_cost', 0.0)
            return text, cost
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (retry {attempt+1}/{max_retries}): {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  API error (final failure): {e}")
                return '', 0.0
    return '', 0.0


def run_inference(
    df: pd.DataFrame,
    mode: str = 'zero_shot',
    model: str = 'gpt-4o-mini',
    workers: int = 4,
) -> pd.DataFrame:
    """
    Run parallel batch inference using a thread pool.

    Args:
        df:      DataFrame with 'id' and 'comment_text' columns
        mode:    'zero_shot', 'few_shot_5', or 'few_shot_10'
        model:   OpenAI model name
        workers: Number of concurrent API requests

    Returns:
        DataFrame with columns: id, raw_response, pred_<label>...
    """
    assert mode in ('zero_shot', 'few_shot_5', 'few_shot_10'), \
        f"mode must be 'zero_shot', 'few_shot_5', or 'few_shot_10', got: {mode}"

    system    = ZERO_SHOT_SYSTEM if mode == 'zero_shot' else FEW_SHOT_SYSTEM
    prompt_fn = {
        'zero_shot':   zero_shot_user,
        'few_shot_5':  few_shot_5_user,
        'few_shot_10': few_shot_10_user,
    }[mode]
    total     = len(df)
    completed = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, row in df.iterrows():
            future = executor.submit(call_llm, system, prompt_fn(row['comment_text']), model)
            futures[future] = row

        records_map = {}
        total_cost = 0.0

        for future in as_completed(futures):
            row          = futures[future]
            raw, cost    = future.result()
            parsed       = parse_response(raw)
            total_cost  += cost

            record = {'id': row['id'], 'raw_response': raw}
            record.update({f'pred_{k}': v for k, v in parsed.items()})
            records_map[row.name] = record

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")

    print(f"  Estimated cost: ${total_cost:.6f} USD")
    records = [records_map[i] for i in sorted(records_map)]
    return pd.DataFrame(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM inference')
    parser.add_argument('--mode',    choices=['zero_shot', 'few_shot_5', 'few_shot_10'], default='zero_shot')
    parser.add_argument('--model',   default='gpt-4o-mini')
    parser.add_argument('--input',   default='results/sample.csv')
    parser.add_argument('--output',  default=None)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    output_path = args.output or f'results/predictions_{args.mode}.csv'

    df = pd.read_csv(args.input)
    print(f"Starting inference: {len(df)} comments | mode={args.mode} | model={args.model}")

    preds = run_inference(df, mode=args.mode, model=args.model, workers=args.workers)
    preds.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
