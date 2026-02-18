from app import summarizer


def test_summarize_text_word_limit():
    text = ("This is sentence one. " * 80).strip()
    out = summarizer.summarize_text(text, max_words=20)
    assert isinstance(out, str)
    assert len(out.split()) <= 20


def test_summarizer_fallback_for_empty():
    assert summarizer.summarize_text("", max_words=10) == ""