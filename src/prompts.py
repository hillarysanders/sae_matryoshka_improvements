# prompts.py
from __future__ import annotations

from typing import Dict, List


def load_prompt_sets() -> Dict[str, List[str]]:
    """Small, curated prompt sets for later concept-fragmentation evaluation.

    Keep these tiny at first; expand once the training loop is stable.
    """
    return {
        "starts_with_e": [
            "elephant",
            "elegant",
            "energy",
            "envelope",
            "every",
            "earth",
            "eager",
            "exercise",
            "echo",
            "europe",
        ],
        "elephant_contexts": [
            "The elephant walked slowly across the savannah.",
            "An elephant is the largest land animal.",
            "We saw an elephant at the zoo today.",
            "Elephants have remarkable memories.",
            "The elephant used its trunk to pick up food.",
        ],
        "hebrew_snippets": [
            "שלום! מה שלומך?",
            "אני אוהב ללמוד עברית.",
            "זה מבחן קצר של טקסט בעברית.",
        ],
        "punctuation": [
            "Hello, world!",
            "Wait... what?",
            "Is this (really) necessary?",
            "Quotes: 'single' and \"double\".",
        ],
        "numbers": [
            "The answer is 42.",
            "In 2025, many models got larger.",
            "Version 2.0 was a big release.",
        ],
    }
