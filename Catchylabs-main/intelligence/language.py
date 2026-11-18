"""Conservative language and script hints for English, Hindi, and Hinglish."""

from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z]+|[\u0900-\u097f]+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_ROMAN_HINDI = {
    "aap", "accha", "acha", "apne", "aur", "bahut", "bana", "banana",
    "banani", "banao", "bhai", "chahiye", "hai", "hain", "ka", "kahani",
    "ke", "ki", "ko", "kya", "lage", "lekin", "liye", "main", "mera",
    "meri", "mujhe", "nahi", "nahin", "naye", "par", "raha", "rahi",
    "se", "toh", "tum", "wala", "wali", "yeh", "ye", "iska", "isko",
    "karna", "karo", "kuch",
}
_ENGLISH_MARKERS = {
    "about", "campaign", "clear", "concept", "core", "competitor", "funny",
    "feel", "hook", "message", "reel", "script", "but", "ad",
}


def detect_language(text: str) -> dict[str, Any]:
    """Return a heuristic label and confidence; this is not identity inference."""
    words = [word.lower() for word in _WORD_RE.findall(text or "")]
    latin_words = [word for word in words if word.isascii()]
    devanagari_count = len(_DEVANAGARI_RE.findall(text or ""))
    letter_count = sum(character.isalpha() for character in text or "")
    hindi_hits = sum(word in _ROMAN_HINDI for word in latin_words)
    english_hits = sum(word in _ENGLISH_MARKERS for word in latin_words)

    if devanagari_count and latin_words:
        language = "hinglish" if english_hits else "hindi"
        script = "mixed"
        confidence = 0.9 if language == "hinglish" else 0.82
    elif devanagari_count:
        language = "hindi"
        script = "devanagari"
        confidence = min(0.99, 0.72 + (devanagari_count / max(letter_count, 1)) * 0.27)
    elif hindi_hits >= 2 and english_hits:
        language = "hinglish"
        script = "latin"
        confidence = min(0.92, 0.65 + 0.06 * (hindi_hits + english_hits))
    elif hindi_hits >= 2:
        language = "hindi"
        script = "latin"
        confidence = min(0.88, 0.62 + 0.06 * hindi_hits)
    else:
        language = "english"
        script = "latin" if latin_words else "unknown"
        confidence = 0.72 if latin_words else 0.4

    return {
        "language": language,
        "script": script,
        "confidence": round(confidence, 2),
        "method": "heuristic_script_and_lexicon",
        "limitations": "Indic code-switching and Romanized Hindi detection is approximate.",
    }


def resolve_language(text: str, preference: str = "auto") -> dict[str, Any]:
    """Apply a supported explicit preference or conservatively detect the input."""
    if preference not in {"auto", "english", "hindi", "hinglish"}:
        raise ValueError("language preference must be auto, english, hindi, or hinglish")
    detected = detect_language(text)
    language = detected["language"] if preference == "auto" else preference
    return {
        **detected,
        "language": language,
        "preference": preference,
        "selection_method": "automatic_detection" if preference == "auto" else "user_preference",
    }


def output_language_instruction(language: str) -> str:
    """Provide explicit generation guidance without changing evidence or identifiers."""
    instructions = {
        "english": "Respond in clear, natural English.",
        "hindi": (
            "Respond in natural, conversational Hindi using Devanagari script. "
            "Avoid unnecessarily formal or machine-translated phrasing."
        ),
        "hinglish": (
            "Respond in natural conversational Hinglish using Roman script. "
            "Code-switch only where it sounds natural; do not mechanically alternate languages."
        ),
    }
    if language not in instructions:
        raise ValueError("output language must be english, hindi, or hinglish")
    return (
        f"{instructions[language]} Preserve brand names, product names, URLs, model IDs, "
        "and technical identifiers as written. Do not invent cultural or audience facts."
    )
