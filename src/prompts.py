"""
Prompt templates and response parser module.

- Builds zero-shot and few-shot prompts for toxicity classification
- Supports 5-example and 10-example few-shot variants
- Parses LLM output into a binary label dictionary
"""

import json
import os
import re

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

ZERO_SHOT_SYSTEM = (
    "You are a content moderation assistant. "
    "Classify the given comment using these toxicity labels: "
    "toxic, severe_toxic, obscene, threat, insult, identity_hate. "
    "Output ONLY the applicable labels, comma-separated. "
    "If none apply, output exactly: none. "
    "Do not explain or add any other text."
)


def zero_shot_user(comment: str) -> str:
    return f'Comment: "{comment}"'


# ---------------------------------------------------------------------------
# Few-shot (shared system prompt)
# ---------------------------------------------------------------------------

FEW_SHOT_SYSTEM = (
    "You are a content moderation assistant. "
    "Classify the given comment using these toxicity labels: "
    "toxic, severe_toxic, obscene, threat, insult, identity_hate. "
    "Output ONLY the applicable labels, comma-separated. "
    "If none apply, output exactly: none. "
    "Do not explain or add any other text."
)

# Synthetic fallback examples (used when few_shot_examples.json has not been built yet)
_SYNTHETIC_FEW_SHOT_EXAMPLES = [
    # ── 5-example set ──────────────────────────────────────────────────────
    {"comment": "You are a complete moron and nobody likes you.",          "labels": "toxic, insult"},
    {"comment": "I know where you live. You better watch your back.",      "labels": "toxic, threat"},
    {"comment": "This is absolute garbage!! You disgusting piece of filth!!", "labels": "toxic, severe_toxic, obscene, insult"},
    {"comment": "Go back to your country, you don't belong here.",         "labels": "toxic, identity_hate"},
    {"comment": "It was fucking cool", "labels": "none"},
    # ── Additional 5 examples (used in 10-example set only) ────────────────
    {"comment": "I will find you and make you pay for this.",              "labels": "toxic, threat"},
    {"comment": "Your kind has no place in civilised society.",            "labels": "toxic, identity_hate"},
    {"comment": "This f***ing idiot has no clue what he's talking about.", "labels": "toxic, obscene, insult"},
    {"comment": "What a great article! Well researched and informative.",   "labels": "none"},
    {"comment": "You worthless piece of garbage, you should be ashamed of yourself.", "labels": "toxic, severe_toxic, insult"},
]

_FEW_SHOT_EXAMPLES_PATH = 'results/few_shot_examples.json'


def _load_few_shot_examples() -> list[dict]:
    if os.path.exists(_FEW_SHOT_EXAMPLES_PATH):
        with open(_FEW_SHOT_EXAMPLES_PATH, encoding='utf-8') as f:
            return json.load(f)
    return _SYNTHETIC_FEW_SHOT_EXAMPLES


_FEW_SHOT_EXAMPLES = _load_few_shot_examples()


def _build_block(examples: list) -> str:
    return "\n\n".join(
        f'Comment: "{ex["comment"]}"\nLabels: {ex["labels"]}'
        for ex in examples
    )


# Real-data blocks (from results/few_shot_examples.json)
_FEW_SHOT_5_BLOCK  = _build_block(_FEW_SHOT_EXAMPLES[:5])
_FEW_SHOT_10_BLOCK = _build_block(_FEW_SHOT_EXAMPLES[:10])

# Synthetic blocks (hardcoded LLM-generated examples)
_FEW_SHOT_5_SYNTH_BLOCK  = _build_block(_SYNTHETIC_FEW_SHOT_EXAMPLES[:5])
_FEW_SHOT_10_SYNTH_BLOCK = _build_block(_SYNTHETIC_FEW_SHOT_EXAMPLES[:10])


def few_shot_5_user(comment: str) -> str:
    return f'{_FEW_SHOT_5_BLOCK}\n\nComment: "{comment}"\nLabels:'


def few_shot_10_user(comment: str) -> str:
    return f'{_FEW_SHOT_10_BLOCK}\n\nComment: "{comment}"\nLabels:'


def few_shot_5_synth_user(comment: str) -> str:
    return f'{_FEW_SHOT_5_SYNTH_BLOCK}\n\nComment: "{comment}"\nLabels:'


def few_shot_10_synth_user(comment: str) -> str:
    return f'{_FEW_SHOT_10_SYNTH_BLOCK}\n\nComment: "{comment}"\nLabels:'


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(response: str) -> dict:
    """
    Convert a raw LLM response string into a {label: 0/1} dictionary.

    - 'none' or empty response → all labels set to 0
    - Comma- or semicolon-separated labels → matching labels set to 1
    - Unrecognised tokens are silently ignored
    """
    result = {label: 0 for label in LABELS}

    cleaned = response.strip().lower()
    if not cleaned or cleaned == 'none':
        return result

    for token in re.split(r'[,;]', cleaned):
        token = token.strip()
        if token in result:
            result[token] = 1

    return result
