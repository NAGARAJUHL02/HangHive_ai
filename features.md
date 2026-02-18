# HangHive AI — Features

Core features implemented in this workspace

- AI chatbots for different communities (`community_type` controls tone and behavior)
- Reply behavior: responds to any user message by default; optional `require_mention` to require `@AI`
- Automod AI
  - Spam detection (links, repeated posts, gibberish)
  - Suspicious-activity detection (repeated content, mention/URL density)
  - Toxicity detection (HF model + heuristic fallback)
  - Unsafe-content blocking (self-harm, threats, PII heuristics)
- Conversation summarization (chunked HF summarizer, word-limited)
- Moderation logging + simple SQLite DB + admin endpoints (`/admin/moderation`)
- Terminal chatbot for quick local testing
- Clean, modular FastAPI backend with Pydantic models and unit tests

How to run

- Start API: `uvicorn app.main:app --reload`
- Terminal chatbot: `python -m app.terminal_chatbot`
- Run tests: `pytest -q`

Examples

- Chat (curl):
  curl -X POST http://127.0.0.1:8001/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"How do I install the app?","community_type":"developers"}'

- See moderation events:
  GET http://127.0.0.1:8001/admin/moderation

Next improvements you can ask for

- Add a web admin dashboard to surface moderation events
- Integrate stronger safety / commercial moderation APIs
- Rate limiting, metrics, and streaming responses
- CI pipeline + model caching on GPU

Tell me which improvement to prioritize and I'll implement it next.

