from __future__ import annotations

import re

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def detect_language(user_query: str, explicit_language: str | None = None) -> str:
    if explicit_language:
        return explicit_language.lower()

    text = user_query.strip()
    lower_text = text.lower()

    if DEVANAGARI_PATTERN.search(text):
        return "hindi"

    hindi_markers = [
        "kya", "kaise", "kyun", "ka", "ki", "ke", "mujhe", "samjhao",
        "batao", "agar", "nahi", "hai", "raha", "rahe", "karna", "chahiye",
    ]
    english_markers = [
        "what", "how", "why", "explain", "summarize", "compare", "give", "show",
    ]

    words = lower_text.split()
    has_hindi_marker = any(word in words for word in hindi_markers)
    has_english_marker = any(word in words for word in english_markers)

    if has_hindi_marker and has_english_marker:
        return "hinglish"
    if has_hindi_marker:
        return "hinglish"
    if has_english_marker:
        return "english"

    return "hinglish"