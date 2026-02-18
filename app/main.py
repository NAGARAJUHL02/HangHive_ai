from typing import List, Optional
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .chatbot import generate_reply
from .automod import is_spam, is_toxic, is_suspicious, is_unsafe
from .summarizer import summarize_text
from .moderation import init_db, log_event, get_events


app = FastAPI(title="HangHive AI — Community Assistant")

# initialize moderation DB
init_db()


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    context: Optional[List[object]] = None
    # optional community type to adapt assistant style (e.g. "developers", "gaming")
    community_type: Optional[str] = "general"
    # when true the endpoint will only respond if the message contains @AI
    require_mention: Optional[bool] = False


class ChatResponse(BaseModel):
    handled: bool
    blocked: bool = False
    reason: Optional[str] = None
    reply: Optional[str] = None


class SummarizeRequest(BaseModel):
    conversation: str
    # max summary length in words (default 60 to keep summaries concise)
    max_length: Optional[int] = 60


class SummarizeResponse(BaseModel):
    summary: str


def contains_ai_mention(text: str) -> bool:
    """Return True if the text contains an @AI mention (word-boundary, case-insensitive)."""
    return bool(re.search(r'(^|\s)@AI\b', text, flags=re.IGNORECASE))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Handle incoming chat messages.

    - If message does not mention @AI -> handled=False and no reply is generated.
    - If @AI is present -> run automod (spam/toxicity). If clean, generate reply.
    """
    # honor optional require_mention flag (default: reply to any message)
    if req.require_mention and not contains_ai_mention(req.message):
        return ChatResponse(handled=False)

    # Run automod checks before invoking the chatbot
    try:
        if is_spam(req.message):
            log_event("blocked", req.user_id, req.message, reason="spam", metadata={"community_type": req.community_type})
            return ChatResponse(handled=True, blocked=True, reason="spam")

        if is_suspicious(req.message, recent_messages=(req.context or [])):
            log_event("blocked", req.user_id, req.message, reason="suspicious", metadata={"community_type": req.community_type})
            return ChatResponse(handled=True, blocked=True, reason="suspicious")

        if is_unsafe(req.message):
            log_event("blocked", req.user_id, req.message, reason="unsafe", metadata={"community_type": req.community_type})
            return ChatResponse(handled=True, blocked=True, reason="unsafe")

        if is_toxic(req.message):
            log_event("blocked", req.user_id, req.message, reason="toxic", metadata={"community_type": req.community_type})
            return ChatResponse(handled=True, blocked=True, reason="toxic")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"automod error: {e}")

    # Generate AI reply using chatbot module (passes community_type + context)
    try:
        reply = generate_reply(
            req.message,
            community_type=(req.community_type or "general"),
            context=(req.context or []),
        )
        # log successful reply for moderation/analytics
        log_event("reply", req.user_id, req.message, reason=None, metadata={"community_type": req.community_type, "reply_preview": (reply or '')[:200]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chatbot error: {e}")

    return ChatResponse(handled=True, blocked=False, reply=reply)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(req: SummarizeRequest):
    """Return a short summary for the provided conversation text."""
    try:
        summary = summarize_text(req.conversation, max_words=(req.max_length or 60))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"summarizer error: {e}")

    return SummarizeResponse(summary=summary)


# --- Admin / moderation endpoints -------------------------------------------------
@app.get("/admin/moderation")
def admin_get_moderation(limit: int = 100):
    """Return recent moderation events (no auth for now)."""
    events = get_events(limit)
    return {"count": len(events), "events": events}


@app.get("/admin/moderation/{event_id}")
def admin_get_moderation_event(event_id: int):
    events = get_events(limit=1000)
    for e in events:
        if e["id"] == event_id:
            return e
    raise HTTPException(status_code=404, detail="event not found")


@app.get("/")
def root():
    return {"service": "HangHive AI — Community Assistant", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
