from __future__ import annotations

import math
import re
from typing import Any


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Split on sentence boundaries while keeping punctuation
        # Using positive lookbehind to avoid stripping punctuation
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        chunks = self._split(text, self.separators)
        return [c for c in chunks if c]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            # Fallback: slice directly by chunk_size
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        if sep not in current_text:
            return self._split(current_text, next_seps)

        raw_pieces = current_text.split(sep)
        pieces: list[str] = []
        for p in raw_pieces:
            if not p:
                continue
            if len(p) > self.chunk_size:
                pieces.extend(self._split(p, next_seps))
            else:
                pieces.append(p)

        # Merge small pieces back up towards chunk_size
        merged_chunks: list[str] = []
        current_chunk = ""
        for p in pieces:
            if not current_chunk:
                current_chunk = p
            else:
                candidate = f"{current_chunk}{sep}{p}"
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    merged_chunks.append(current_chunk)
                    current_chunk = p
        if current_chunk:
            merged_chunks.append(current_chunk)
        return merged_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_prod / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=max(1, int(chunk_size * 0.1)))
        by_sentences = SentenceChunker(max_sentences_per_chunk=3)
        recursive = RecursiveChunker(chunk_size=chunk_size)

        strategies = {
            "fixed_size": fixed.chunk(text),
            "by_sentences": by_sentences.chunk(text),
            "recursive": recursive.chunk(text),
        }

        results: dict[str, Any] = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count > 0 else 0.0
            results[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return results


class HeadingChunker:
    """
    Split text by Markdown headings (#, ##, ###) or regulation articles (Điều X:).

    For sections exceeding max_chunk_size, recursively sub-splits and injects
    the parent heading context into each sub-chunk to preserve provenance.
    """

    def __init__(self, max_chunk_size: int = 500) -> None:
        self.max_chunk_size = max_chunk_size
        self._recursive_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        heading_pattern = r"(?=(?:^|\n)#{1,6}\s+|(?:^|\n)Điều\s+\d+[:\.])"
        sections = [s.strip() for s in re.split(heading_pattern, text.strip()) if s.strip()]

        if not sections:
            return []

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                first_line, _, rest = section.partition("\n")
                heading_title = first_line.strip() if first_line.startswith(("#", "Điều")) else ""

                sub_text = rest.strip() if heading_title else section
                sub_chunks = self._recursive_chunker.chunk(sub_text)

                for sc in sub_chunks:
                    if heading_title and not sc.startswith(heading_title):
                        chunks.append(f"{heading_title}\n{sc}")
                    else:
                        chunks.append(sc)

        return chunks



