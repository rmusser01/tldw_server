from tldw_Server_API.app.api.v1.API_Deps import personalization_deps


class _FailingPersonalizationDB:
    def insert_usage_event(self, event):
        raise RuntimeError("usage backend exploded /tmp/secret")


def test_usage_event_logger_fail_open_logs_safe_message():
    messages: list[str] = []
    sink_id = personalization_deps.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="DEBUG",
        format="{message}",
    )
    try:
        result = personalization_deps.UsageEventLogger(
            user_id="user-1",
            db=_FailingPersonalizationDB(),
        ).log_event("media.view", resource_id="resource-1")
    finally:
        personalization_deps.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert result is None
    assert "UsageEventLogger failed (non-fatal)" in rendered_logs
    assert "usage backend exploded" not in rendered_logs
    assert "/tmp/secret" not in rendered_logs
