from app import moderation


def test_log_and_get_event():
    eid = moderation.log_event("test", "user123", "hello", reason="unit-test", metadata={"a":1})
    assert isinstance(eid, int)
    events = moderation.get_events(limit=5)
    assert any(e["id"] == eid for e in events)
