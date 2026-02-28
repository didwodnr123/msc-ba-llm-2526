"""
Prompt templates and response parser module.

- Builds zero-shot and few-shot prompts for toxicity classification
- Parses LLM output into a binary label dictionary
"""

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
# Few-shot
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = [
    {
        "comment": "You are a complete moron and nobody likes you.",
        "labels": "toxic, insult",
    },
    {
        "comment": "I know where you live. You better watch your back.",
        "labels": "toxic, threat",
    },
    {
        "comment": "This is absolute garbage!! You disgusting piece of filth!!",
        "labels": "toxic, severe_toxic, obscene, insult",
    },
    {
        "comment": "Go back to your country, you don't belong here.",
        "labels": "toxic, identity_hate",
    },
    {
        "comment": "Thank you for the clarification. I appreciate your help.",
        "labels": "none",
    },
]

FEW_SHOT_SYSTEM = (
    "You are a content moderation assistant. "
    "Classify the given comment using these toxicity labels: "
    "toxic, severe_toxic, obscene, threat, insult, identity_hate. "
    "Output ONLY the applicable labels, comma-separated. "
    "If none apply, output exactly: none. "
    "Do not explain or add any other text."
)

_FEW_SHOT_BLOCK = "\n\n".join(
    f'Comment: "{ex["comment"]}"\nLabels: {ex["labels"]}'
    for ex in _FEW_SHOT_EXAMPLES
)


def few_shot_user(comment: str) -> str:
    return f"{_FEW_SHOT_BLOCK}\n\nComment: \"{comment}\"\nLabels:"


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
