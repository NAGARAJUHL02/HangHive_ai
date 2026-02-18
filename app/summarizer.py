import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    try:
        from transformers import pipeline

        _summarizer = pipeline("summarization", model=_SUMMARIZER_MODEL, device=-1)
        return _summarizer
    except Exception as exc:
        _summarizer = None
        logger.debug("summarizer unavailable: %s", exc)
        raise


def _chunk_text(text: str, max_chunk_chars: int = 2000) -> List[str]:
    """Split text into chunks that are safe for the summarization model."""
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[str] = []
    current = []
    curr_len = 0
    for s in sentences:
        if curr_len + len(s) + 1 > max_chunk_chars and current:
            chunks.append(" ".join(current))
            current = [s]
            curr_len = len(s)
        else:
            current.append(s)
            curr_len += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text: str, max_words: int = 60) -> str:
    """Summarize `text` using HF summarization pipeline and return clean text.

    - Limits the returned summary to `max_words` words.
    - Handles long input by chunking + combining.
    - Returns a short, readable string even on internal errors.
    """
    if not text or not text.strip():
        return ""

    try:
        summarizer = _get_summarizer()

        chunks = _chunk_text(text, max_chunk_chars=2000)
        if not chunks:
            return ""

        partials = []
        for chunk in chunks:
            out = summarizer(chunk, max_length=130, min_length=20, do_sample=False, truncation=True)
            if isinstance(out, list) and out:
                partials.append(out[0].get("summary_text", "").strip())

        combined = " ".join(partials)
        # if there were multiple partials, compress once more
        if len(partials) > 1:
            out = summarizer(combined, max_length=130, min_length=30, do_sample=False, truncation=True)
            if isinstance(out, list) and out:
                combined = out[0].get("summary_text", "").strip()

        # enforce max_words limit
        words = combined.split()
        if len(words) > max_words:
            combined = " ".join(words[:max_words]).rstrip() + "..."

        return combined.strip()
    except Exception:
        logger.exception("summarizer pipeline failed, falling back to extractive summary")
        # fallback: return first max_words words of the text
        words = re.findall(r"\S+", text)
        if not words:
            return ""
        return " ".join(words[:max_words]).rstrip() + ("..." if len(words) > max_words else "")


# Backwards-compatible name used by the FastAPI handlers
def summarize_conversation(text: str, max_length: int = 60) -> str:
    # treat max_length as max words for compatibility with the new summarizer
    return summarize_text(text, max_words=(max_length or 60))