"""Utility helpers for deterministic embeddings, scoring, and IDs."""

from __future__ import annotations

import hashlib
import math
import re
from statistics import fmean

WORD_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization for simple semantic matching."""

    normalized = text.lower().replace("_", " ").replace("-", " ")
    return WORD_RE.findall(normalized)


def stable_id(prefix: str, *parts: object, length: int = 10) -> str:
    """Create a deterministic short ID from arbitrary values."""

    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def deterministic_vector(seed: str, dim: int = 16) -> tuple[float, ...]:
    """Create a deterministic pseudo-embedding from text seed."""

    chunks: list[float] = []
    current = seed
    while len(chunks) < dim:
        digest = hashlib.sha256(current.encode("utf-8")).digest()
        for idx in range(0, len(digest), 2):
            if len(chunks) >= dim:
                break
            value = int.from_bytes(digest[idx : idx + 2], "big")
            # Map to [-1, 1]
            chunks.append((value / 32767.5) - 1.0)
        current = current + "#"

    norm = math.sqrt(sum(value * value for value in chunks)) or 1.0
    return tuple(value / norm for value in chunks)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity for equal-length vectors."""

    if len(left) != len(right):
        raise ValueError("vector lengths must match")
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
    right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
    return dot / (left_norm * right_norm)


def overlap_score(query_tokens: list[str], semantic_tokens: list[str]) -> float:
    """Token overlap score bounded to [0,1]."""

    if not query_tokens:
        return 0.0
    query = set(query_tokens)
    semantic = set(semantic_tokens)
    return len(query & semantic) / len(query)


def mean(values: list[float]) -> float:
    """Robust arithmetic mean with empty-list protection."""

    if not values:
        return 0.0
    return float(fmean(values))


def smooth_curve(values: list[float], window: int) -> list[float]:
    """Simple moving-average smoothing used for temporal localization."""

    if window <= 1 or len(values) <= 1:
        return values[:]

    half_window = max(1, window // 2)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        smoothed.append(mean(values[start:end]))
    return smoothed
