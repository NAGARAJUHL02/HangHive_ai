import pytest
from app import automod


def test_is_spam_links():
    assert automod.is_spam("https://a.com https://b.com https://c.com")


def test_is_suspicious_repeats():
    recent = ["hi", "hi", "hi"]
    assert automod.is_suspicious("hi", recent_messages=recent)


def test_is_unsafe_keyword():
    assert automod.is_unsafe("I will kill you")


def test_check_toxicity_heuristic():
    label, score = automod.check_toxicity("You are an idiot")
    assert label is not None
