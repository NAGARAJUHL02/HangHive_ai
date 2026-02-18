"""Terminal-based AI chatbot for HangHive.

Run from workspace root:
    python -m app.terminal_chatbot

Behaviour:
- Interactive loop until the user types "exit" or presses Ctrl-C
- Validates community type on startup (invalid → defaults to "general")
- Pre-warms both chat and moderation models for fast first reply
- Runs automod checks (spam / suspicious / toxicity) before generating
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List

# Suppress ALL HuggingFace noise before any other app import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

# import with fallback so script is runnable both as a module and directly
try:
    from app.automod import detect_spam, check_toxicity, is_suspicious, is_unsafe
    from app.chatbot import generate_reply, warm_up_model, VALID_COMMUNITY_TYPES
    from app.automod import warm_up_automod
    from app.moderation import log_event
except Exception:
    from automod import detect_spam, check_toxicity, is_suspicious, is_unsafe, warm_up_automod  # type: ignore
    from chatbot import generate_reply, warm_up_model, VALID_COMMUNITY_TYPES  # type: ignore
    from moderation import log_event  # type: ignore


WELCOME = """
╔══════════════════════════════════════════════╗
║   HangHive AI — Community Assistant          ║
║   Type messages and press Enter.             ║
║   Type "exit" to quit.                       ║
╚══════════════════════════════════════════════╝
"""


def choose_community_type() -> str:
    """Prompt user to pick a community type; defaults to 'general' on bad input."""
    options = sorted(VALID_COMMUNITY_TYPES)
    print(f"  Available community types: {', '.join(options)}")
    choice = input("  Select community type (Enter = general): ").strip().lower()
    if not choice:
        return "general"
    if choice not in VALID_COMMUNITY_TYPES:
        print(f'  ⚠ "{choice}" is not a valid type — defaulting to "general".')
        return "general"
    return choice


def _is_model_toxic(label: str, score: float) -> bool:
    if not label:
        return False
    low = label.lower()
    indicators = ("tox", "abuse", "offens", "insult", "threat", "hate")
    return any(sub in low for sub in indicators) and score >= 0.6


def main() -> None:
    print(WELCOME)
    community_type = choose_community_type()
    print(f"  Using community type: {community_type}\n")

    # ---- warm up models ----------------------------------------------------
    print("  Loading models …", end=" ", flush=True)
    chat_ok = warm_up_model()
    mod_ok = warm_up_automod()
    if chat_ok and mod_ok:
        print("ready ✓\n")
    elif chat_ok:
        print("chat ready ✓  (moderation model unavailable)\n")
    else:
        print("models unavailable — replies may be slow or fallback-only.\n")

    # ---- conversation loop -------------------------------------------------
    history: List[Dict[str, Any]] = []
    MAX_HISTORY = 10  # keep last N turns

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1) Spam
        try:
            if detect_spam(user_input):
                print("Bot: ⛔ Spam detected — message blocked.\n")
                try:
                    log_event("blocked", None, user_input, reason="spam",
                              metadata={"community_type": community_type})
                except Exception:
                    pass
                continue
        except Exception:
            pass

        # 2) Suspicious activity
        try:
            recent = [h["content"] for h in history if h.get("role") == "user"]
            if is_suspicious(user_input, recent_messages=recent):
                print("Bot: ⚠ Suspicious activity detected — message blocked.\n")
                try:
                    log_event("blocked", None, user_input, reason="suspicious",
                              metadata={"community_type": community_type})
                except Exception:
                    pass
                continue
        except Exception:
            pass

        # 3) Toxicity / unsafe
        try:
            if is_unsafe(user_input):
                print("Bot: 🚫 Unsafe content detected — please be respectful.\n")
                try:
                    log_event("blocked", None, user_input, reason="unsafe",
                              metadata={"community_type": community_type})
                except Exception:
                    pass
                continue
            label, score = check_toxicity(user_input)
            if _is_model_toxic(label, score):
                print("Bot: 🚫 Toxic content detected — please be respectful.\n")
                try:
                    log_event("blocked", None, user_input, reason="toxic",
                              metadata={"community_type": community_type,
                                        "tox_label": label, "tox_score": score})
                except Exception:
                    pass
                continue
        except Exception:
            pass  # allow generation even if automod fails

        # 4) Generate reply
        try:
            reply = generate_reply(
                user_input,
                community_type=community_type,
                context=history,
            )
        except Exception as exc:
            print("Bot: Error generating reply — try again later.\n")
            traceback.print_exception(exc, file=sys.stderr)
            continue

        # Maintain conversation history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()