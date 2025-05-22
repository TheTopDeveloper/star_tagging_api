# app/utils.py

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Basic cleaning: remove excessive whitespace, URLs, and special characters.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    return text

def filter_keywords(keywords: List[str], max_keywords: int = 15) -> List[str]:
    """
    Deduplicate and trim list of keywords.
    """
    seen = set()
    filtered = []
    for keyword in keywords:
        kw = keyword.strip().lower()
        if kw and kw not in seen:
            seen.add(kw)
            filtered.append(kw)
        if len(filtered) == max_keywords:
            break
    return filtered

def is_valid_input(text: str) -> bool:
    """
    Very basic content check: ensure there's enough content to extract keywords.
    """
    return len(text.strip()) > 30
