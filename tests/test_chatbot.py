from app import chatbot


class DummyGen:
    def __call__(self, prompt, **kwargs):
        return [{"generated_text": prompt + " Hello from dummy."}]


def test_generate_reply_with_context(monkeypatch):
    monkeypatch.setattr(chatbot, "_get_text_generator", lambda: DummyGen())
    reply = chatbot.generate_reply("@AI how are you?", community_type="general", context=["previous message"])
    assert "Hello from dummy" in reply
    assert isinstance(reply, str)