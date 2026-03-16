# Vibe Coding Prompts

## Phase 1: Project Setup & Planning
- Create a project plan for a multi-label toxic comment classification system using LLMs and prompt engineering
- Task: classify comments into 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate) using the Jigsaw dataset
- Inference only via OpenAI API — no fine-tuning
- Outline dataset, evaluation criteria, implementation schedule, and project structure

## Phase 2: Data Preprocessing
- Load data/train.csv, clean comment text (remove HTML tags, normalise whitespace)
- Stratified sampling: 50% toxic / 50% non-toxic, n_samples configurable, default 500
- A comment is toxic if any of the 6 labels is 1
- Save sampled output to results/sample.csv
- Add load_test_set() to load test.csv + test_labels.csv, merge on id, drop unlabelled rows (toxic == -1)
- Apply the same stratified sampling logic to the test set

## Phase 3: Prompt Templates
- Write a zero-shot system prompt instructing the model to output only comma-separated labels or "none"
- Write zero_shot_user(comment) to format a single comment as the user message
- Write few_shot_5_user and few_shot_10_user that prepend labelled examples before the target comment
- Write parse_response(response) to convert raw LLM output to a {label: 0/1} dict — handle "none", empty string, comma/semicolon-separated labels, silently ignore unrecognised tokens
- Add synthetic few-shot variants (few_shot_5_synth, few_shot_10_synth) using hardcoded LLM-generated examples instead of real training data

## Phase 4: Inference Pipeline
- Call the OpenAI Chat Completions API with system + user messages
- Use max_completion_tokens=30, temperature=0 for deterministic output
- Implement exponential back-off retry with 3 attempts; use longer waits for 429 rate-limit errors
- Parallelise requests using ThreadPoolExecutor with configurable workers, default 4
- Print progress every 50 samples
- Return a DataFrame with id, raw_response, and pred_{label} columns
- Support all 5 modes: zero_shot, few_shot_5, few_shot_10, few_shot_5_synth, few_shot_10_synth
- Add provider routing: use Groq-compatible client for open-source model names (llama, mixtral, gemma, etc.), standard OpenAI client otherwise

## Phase 5: Few-Shot Example Builder
- Extract real labelled examples from train.csv for use in few-shot prompts
- Select one example per target label combination (10 combinations covering none, single-label, and multi-label cases)
- Prefer short comments (≤150 chars) to keep prompt length manageable
- Exclude any IDs present in the evaluation sample to prevent data leakage
- Save as results/few_shot_examples.json

## Phase 6: Detoxify Baselines
- Run fine-tuned BERT/RoBERTa baselines locally using the detoxify library
- Support two models: toxic-bert (original) and unbiased-toxic-roberta (unbiased)
- Map detoxify output keys to Jigsaw label names
- Apply a 0.5 binary threshold; batch_size=256 for efficiency
- Return the same pred_{label} DataFrame format as LLM inference

## Phase 7: Evaluation Module
- Compute Micro F1 (primary), Macro F1, and Exact Match Accuracy
- Compute per-label precision, recall, F1 using sklearn
- Treat undefined precision/recall as 0 (zero_division=0)
- Print a formatted table and save results/evaluation_summary.csv

## Phase 8: Pipeline Entry Point
- Single entry point run.py for the full pipeline
- Accept --step (preprocess, infer, detoxify, evaluate, all), --mode, --models, --n_samples, --workers, --dataset
- Auto-discover all results/predictions_*.csv files for evaluation — no hardcoded paths
- Parse model and mode from filenames like predictions_gpt-4.1_zero_shot.csv

## Phase 9: Model Selection & Debugging
- gpt-5-mini is returning near-zero recall with content='' and finish_reason='length' — investigate cause
- Measure how many reasoning tokens the model is using per request
- The model uses 64–704 internal reasoning tokens before producing output; max_completion_tokens=30 is exhausted entirely by reasoning, leaving no tokens for the label output
- Find a fast non-reasoning alternative to replace gpt-5-mini — list available OpenAI models via the API and test the best candidate on a small sample
- Update inference.py: simplify to uniform max_completion_tokens=30 for all models; remove gpt-5-mini special casing
- Replace gpt-5-mini with gpt-4.1-mini across all README tables and cost estimates

## Phase 10: README & Documentation
- Add a Discussion section after Results with key findings and future improvements
- Add an inference time table showing per-mode and full-run times for all models
- Add a "Why non-reasoning models?" section explaining the reasoning token overhead and the gpt-5-mini failure
- Add a Vibe Coding Notes section with practical lessons learnt from AI-assisted development
- Use British English throughout (optimisation, labelled, behaviour, analysed)
