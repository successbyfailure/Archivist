import re
from langdetect import detect
from typing import Tuple


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00a0", " ")
    return text.strip()


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def chunk_text(text: str, size: int, overlap: int) -> Tuple[str, ...]:
    if size <= 0:
        return (text,)
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start = end - overlap if overlap > 0 else end
    return tuple(chunks)
