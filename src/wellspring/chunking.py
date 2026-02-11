from __future__ import annotations

import hashlib
from typing import List

from .schemas import Chunk


def make_chunk_id(source_uri: str, start_offset: int, end_offset: int) -> str:
    raw = f"{source_uri}:{start_offset}:{end_offset}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def chunk_text(text: str, source_uri: str, max_chars: int, overlap: int) -> List[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    chunks: List[Chunk] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk_id = make_chunk_id(source_uri, start, end)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_uri=source_uri,
                start_offset=start,
                end_offset=end,
                text=text[start:end],
            )
        )
        if end >= text_len:
            break
        start = end - overlap
    return chunks
