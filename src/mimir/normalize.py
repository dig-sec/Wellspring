from __future__ import annotations

import re
from typing import Optional

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_entity_name(name: str) -> str:
    name = name.strip()
    name = _WHITESPACE_RE.sub(" ", name)
    return name


def normalize_predicate(predicate: str) -> str:
    predicate = predicate.strip().lower()
    predicate = _NON_ALNUM_RE.sub(" ", predicate)
    predicate = _WHITESPACE_RE.sub("_", predicate).strip("_")
    return predicate


def canonical_entity_key(name: str, entity_type: Optional[str]) -> str:
    normalized = normalize_entity_name(name).lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    type_part = (entity_type or "").strip().lower()
    return f"{normalized}|{type_part}".strip("|")
