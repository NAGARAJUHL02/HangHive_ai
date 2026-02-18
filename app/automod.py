import logging
import os
import re
from typing import Iterable, Tuple

# Silence HuggingFace logging (model-load reports, weight tables, etc.)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()
except Exception:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_SPAM_PATTERNS: Iterable[re.Pattern] = [
    re.compile(r"buy now", re.I),
    re.compile(r"click here", re.I),
    re.compile(r"free money", re.I),
]

_TOXIC_WORDS = {"fuck", "shit", "bitch", "idiot", "kill", "hate"}

# Transformers toxicity model (lazy-loaded)
_TOXICITY_MODEL = "unitary/toxic-bert"
_toxicity_detector = None


def _get_toxicity_detector():
    global _toxicity_detector
    if _toxicity_detector is not None:
        return _toxicity_detector

    try:
        from transformers import pipeline

        _toxicity_detector = pipeline("text-classification", model=_TOXICITY_MODEL, device=-1)
        return _toxicity_detector
    except Exception as exc:
        _toxicity_detector = None
        logger.debug("toxicity detector unavailable: %s", exc)
        raise


def warm_up_automod() -> bool:
    """Initialize the toxicity model lazily."""
    try:
        _get_toxicity_detector()
        return True
    except Exception:
        return False


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://", text, re.I))


def is_spam(text: str) -> bool:
    """Improved spam heuristics:

    - obvious spammy phrase matches
    - too many links
    - repeated lines/messages
    - excessive length
    - repeated characters
    """
    if not text:
        return False

    # obvious spammy phrases
    if any(p.search(text) for p in _SPAM_PATTERNS):
        logger.debug("spam detected by pattern")
        return True

    # too many URLs
    urls = re.findall(r"https?://", text, re.I)
    if len(urls) > 2:
        logger.debug("spam detected by too many urls: %d", len(urls))
        return True

    # repeated characters (aaaaaaa)
    if re.search(r"(.)\1{6,}", text):
        logger.debug("spam detected by repeated characters")
        return True

    # repeated lines or repeated short messages
    lines = [l.strip() for l in re.split(r"\n|\r", text) if l.strip()]
    if lines:
        most_common = max(lines, key=lines.count)
        if lines.count(most_common) >= 3:
            logger.debug("spam detected by repeated lines: %s", most_common[:40])
            return True

    # excessive length (likely dump or abuse)
    if len(text) > 8000:
        logger.debug("spam detected by excessive length: %d chars", len(text))
        return True

    return False


def check_toxicity(message: str) -> Tuple[str, float]:
    """Use a HF text-classification pipeline to return (label, score).

    Falls back to a lightweight heuristic when the model isn't available.
    """
    if not message:
        return ("clean", 0.0)

    try:
        detector = _get_toxicity_detector()
        out = detector(message, truncation=True)
        if isinstance(out, list) and out:
            label = out[0].get("label", "unknown")
            score = float(out[0].get("score", 0.0))
        elif isinstance(out, dict):
            label = out.get("label", "unknown")
            score = float(out.get("score", 0.0))
        else:
            label, score = ("unknown", 0.0)

        logger.debug("toxicity model -> %s (%.3f)", label, score)
        return (label, score)
    except Exception:
        # fallback heuristic
        tokens = re.findall(r"\w+", message.lower())
        toxic = any(w in _TOXIC_WORDS for w in tokens)
        return ("heuristic_toxic" if toxic else "clean", 1.0 if toxic else 0.0)


def is_toxic(text: str) -> bool:
    """Public convenience function: try model first, then fallback to blacklist.

    Returns True when message is considered toxic.
    """
    try:
        label, score = check_toxicity(text)
        l = label.lower()
        # treat a few label substrings as toxic indicators
        toxic_indicators = ("tox", "abuse", "offens", "insult", "threat", "hate")
        if any(sub in l for sub in toxic_indicators) and score >= 0.6:
            logger.debug("is_toxic -> model says toxic: %s %.2f", label, score)
            return True
    except Exception:
        logger.debug("toxicity model failed, falling back to heuristic")

    # fallback blacklist
    tokens = re.findall(r"\w+", (text or "").lower())
    return any(w in _TOXIC_WORDS for w in tokens)


# Backwards-compatible alias expected by some callers
def detect_spam(text: str) -> bool:
    """Compatibility wrapper for older code that calls detect_spam()."""
    return is_spam(text)


def is_suspicious(text: str, recent_messages: list[str] | None = None) -> bool:
    """Detect suspicious activity using lightweight heuristics.

    - repeated message content across `recent_messages`
    - excessive URL/mention density
    - unusually short repeated posts
    """
    if not text:
        return False

    # repeated content in recent messages
    if recent_messages:
        matches = sum(1 for m in recent_messages if m.strip() == text.strip())
        if matches >= 3:
            logger.debug("suspicious: repeated message seen %d times", matches)
            return True

    # mention / URL density
    mentions = len(re.findall(r"@\w+", text))
    urls = len(re.findall(r"https?://", text))
    if mentions >= 5 or urls >= 5:
        logger.debug("suspicious: mention/url density mentions=%d urls=%d", mentions, urls)
        return True

    # short but repeating content (e.g., short spammy bursts)
    if len(text) < 40 and re.search(r"(.)\1{4,}", text):
        logger.debug("suspicious: short repeating characters")
        return True

    return False


def is_unsafe(text: str) -> bool:
    """Block obviously unsafe content (self-harm, threats, explicit sexual content).

    This is intentionally conservative — use a proper safety model in production.
    """
    if not text:
        return False

    low = text.lower()

    # quick heuristic checks
    unsafe_indicators = [
        r"\b(i will kill|i'm going to kill|i'm going to hurt)\b",
        r"\b(suicide|kill myself|end my life)\b",
        r"\b(contact me at|my ssn|credit card)\b",
        r"\b(nude|sex with)\b",
    ]
    for pat in unsafe_indicators:
        if re.search(pat, low):
            logger.debug("unsafe content matched pattern: %s", pat)
            return True

    # rely on toxicity model as a final check for threats/hate
    try:
        label, score = check_toxicity(text)
        if isinstance(label, str) and ("threat" in label.lower() or "hate" in label.lower()) and score >= 0.6:
            logger.debug("unsafe content flagged by toxicity model: %s %.2f", label, score)
            return True
    except Exception:
        logger.debug("toxicity model unavailable for is_unsafe check")

    return False