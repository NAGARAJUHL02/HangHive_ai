"""HangHive AI — Text generation module.

Production-ready chatbot reply generation using HuggingFace GPT-2.
Designed for Discord bot integration with clean API surface.

Usage:
    from app.chatbot import generate_reply, warm_up_model, VALID_COMMUNITY_TYPES

    warm_up_model()  # call once at startup
    reply = generate_reply("hello", community_type="general")
"""
from __future__ import annotations

import logging
import os
import re

# ---------------------------------------------------------------------------
# Silence all HuggingFace / tokenizer noise BEFORE any transformers import
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Apply transformers-level silencing (safe even if transformers not installed yet)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "gpt2"
_MAX_NEW_TOKENS = 64  # short replies → faster generation

VALID_COMMUNITY_TYPES = frozenset(
    ["general", "developers", "support", "gaming", "moderation", "study"]
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "developers": "You are a helpful assistant for software developers. You give concise technical advice.",
    "support": "You are a polite support assistant. You help users solve problems step-by-step.",
    "gaming": "You are a friendly gaming assistant. You talk about games and help players.",
    "moderation": "You are a community moderator. You ensure users follow rules and stay respectful.",
    "study": "You are a study assistant. You help students learn and answer academic questions.",
    "general": "You are a helpful community assistant. You give friendly, concise answers.",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _strip_mention(text: str) -> str:
    """Remove @AI mention from message text."""
    return re.sub(r"(?i)@AI\b", "", text).strip()


def _system_prompt_for(community_type: str) -> str:
    """Return the system prompt for *community_type*, defaulting to general."""
    ct = (community_type or "general").lower().strip()
    return _SYSTEM_PROMPTS.get(ct, _SYSTEM_PROMPTS["general"])


def _trim_repetition(text: str) -> str:
    """Detect and remove sentence-level loops in generated text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 3:
        return text

    seen: list[str] = []
    for s in sentences:
        normed = s.strip().lower()
        if normed in [x.strip().lower() for x in seen]:
            # loop detected — return everything up to this point
            break
        seen.append(s)
    return " ".join(seen).strip()


# ---------------------------------------------------------------------------
# Lazy pipeline loader
# ---------------------------------------------------------------------------
_text_generator = None


def _get_text_generator():
    """Lazy-load the HF text-generation pipeline (singleton)."""
    global _text_generator
    if _text_generator is not None:
        return _text_generator

    try:
        from transformers import pipeline

        _text_generator = pipeline(
            "text-generation",
            model=_MODEL_NAME,
            device=-1,  # CPU; set to 0 for CUDA
        )
        # Clear the pipeline's default max_length (50) so it does NOT conflict
        # with max_new_tokens passed at call time.
        _text_generator.model.config.max_length = None
        return _text_generator
    except Exception as exc:
        _text_generator = None
        raise RuntimeError(f"text-generation pipeline unavailable: {exc}") from exc


def warm_up_model() -> bool:
    """Initialise the text-generation pipeline.

    Call once at startup so the first real request is fast.
    Returns True on success, False on failure.
    """
    try:
        _get_text_generator()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_reply(
    message: str,
    community_type: str = "general",
    context: list | None = None,
) -> str:
    """Generate a concise reply to *message*.

    Args:
        message: Raw user message (may contain ``@AI``).
        community_type: Community flavour — must be one of
            :data:`VALID_COMMUNITY_TYPES` (invalid values default to
            ``"general"``).
        context: Optional conversation history.  Each item is either a plain
            string (treated as a user message) or a dict with keys
            ``role`` (``"user"`` / ``"assistant"``) and ``content``.

    Returns:
        The assistant's reply text.  On internal errors a friendly
        fallback string is returned so the caller never sees an exception.
    """
    user_text = _strip_mention(message)
    if not user_text:
        return "Hi — how can I help?"

    # --- build context lines ------------------------------------------------
    history_lines: list[str] = []
    if context:
        for item in context:
            try:
                if isinstance(item, dict):
                    role = (item.get("role") or "user").lower()
                    content = (item.get("content") or "").strip()
                    if not content:
                        continue
                    prefix = "Assistant" if role.startswith("a") else "User"
                    history_lines.append(f"{prefix}: {content}")
                else:
                    text = str(item).strip()
                    if text:
                        history_lines.append(f"User: {text}")
            except Exception:
                continue

    # --- assemble prompt with few-shot examples -----------------------------
    prompt_parts: list[str] = [
        f"System: {_system_prompt_for(community_type)}",
        "Instruction: Answer the user directly. Do not repeat the question or list numbered questions.",
        "",
        "User: hello",
        "Assistant: Hi there! How can I help you today?",
        "User: how are you",
        "Assistant: I'm doing great, thank you! What can I help you with?",
        "",
    ]
    if history_lines:
        prompt_parts.extend(history_lines)

    prompt_parts.append(f"User: {user_text}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    # --- generate -----------------------------------------------------------
    try:
        gen = _get_text_generator()

        # Pass all generation hyper-parameters as direct kwargs.
        # This avoids the "Passing generation_config together with
        # generation-related arguments is deprecated" warning.
        outputs = gen(
            prompt,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=50256,
        )

        if not outputs or not isinstance(outputs, list):
            return "I'm sorry — I couldn't create a response right now."

        raw: str = outputs[0].get("generated_text") or ""

        # --- extract only the new assistant text ----------------------------
        if raw.startswith(prompt):
            reply = raw[len(prompt):].strip()
        else:
            # fallback: strip known prefixes
            reply = raw.split("Assistant:")[-1].strip()

        # strip a leading "Assistant:" the model may echo
        reply = re.sub(r"^\s*Assistant:\s*", "", reply, flags=re.IGNORECASE).strip()

        # cut off at the first new-turn marker
        for marker in ("\nUser:", "\nAssistant:", "\nSystem:"):
            idx = reply.find(marker)
            if idx != -1:
                reply = reply[:idx].strip()

        # collapse whitespace
        reply = re.sub(r"\s{2,}", " ", reply)

        # trim sentence-level repetition
        reply = _trim_repetition(reply)

        # hard limit
        if len(reply) > 500:
            reply = reply[:500].rsplit(".", 1)[0] + "..."

        return reply or "I'm sorry — I couldn't create a response right now."

    except RuntimeError:
        return "Sorry — the text-generation model is not available right now."
    except Exception:
        return "Sorry — I couldn't generate a reply at the moment."