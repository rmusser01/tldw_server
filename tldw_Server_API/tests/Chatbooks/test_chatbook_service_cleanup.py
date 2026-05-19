import asyncio

import pytest

from tldw_Server_API.app.services import chatbooks_cleanup_service as cleanup_module
from tldw_Server_API.tests.Chatbooks.test_chatbook_service import mock_db, service  # noqa: F401


def test_enumerate_user_ids_fail_open_log_omits_raw_base_dir_error(monkeypatch):
    """Base-dir discovery failures should fail open without logging sensitive details."""
    secret_path = "/private/user/dbs?token=raw-cleanup-secret"
    messages: list[str] = []
    sink_id = cleanup_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="DEBUG",
    )

    def _raise_private_error():
        raise RuntimeError(f"cannot access {secret_path}")

    monkeypatch.setattr(
        cleanup_module.DatabasePaths,
        "get_user_db_base_dir",
        _raise_private_error,
    )

    try:
        result = cleanup_module._enumerate_user_ids()
    finally:
        cleanup_module.logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert result == []
    assert "chatbooks_cleanup: failed to resolve user db base dir" in rendered
    assert "cannot access" not in rendered
    assert secret_path not in rendered
    assert "raw-cleanup-secret" not in rendered


@pytest.mark.asyncio
async def test_run_chatbooks_cleanup_loop_sanitizes_outer_noncritical_error(monkeypatch):
    """Outer cleanup-loop failures should log a stable message without raw exception text."""
    secret_path = "/private/chatbooks?token=cleanup-loop-secret"
    messages: list[str] = []
    sleep_calls: list[int] = []
    stop_event = asyncio.Event()
    sink_id = cleanup_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="WARNING",
    )

    def _raise_private_error():
        raise RuntimeError(f"cannot access {secret_path}")

    async def _fake_sleep(seconds: int):
        sleep_calls.append(seconds)
        stop_event.set()

    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "1")
    monkeypatch.setattr(cleanup_module, "_enumerate_user_ids", _raise_private_error)
    monkeypatch.setattr(cleanup_module.asyncio, "sleep", _fake_sleep)

    try:
        await cleanup_module.run_chatbooks_cleanup_loop(stop_event)
    finally:
        cleanup_module.logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert sleep_calls == [1]
    assert "Chatbooks cleanup loop error" in rendered
    assert "cannot access" not in rendered
    assert secret_path not in rendered
    assert "cleanup-loop-secret" not in rendered


def test_cleanup_expired_exports_breaks_on_repeated_no_progress(service, mock_db):
    """Cleanup should stop after repeated batches that cannot mark rows expired."""
    expired_rows = [{"job_id": "job-stuck", "output_path": None}]
    call_counts = {"select": 0, "update": 0}

    def _execute_query(sql, params=None, commit=False):
        if sql.startswith("SELECT * FROM export_jobs"):
            call_counts["select"] += 1
            return expired_rows
        if sql.startswith("UPDATE export_jobs"):
            call_counts["update"] += 1
            raise RuntimeError("update failed")
        return []

    mock_db.execute_query.side_effect = _execute_query

    deleted = service.cleanup_expired_exports(batch_size=1)

    assert deleted == 0
    assert call_counts["select"] == 2
    assert call_counts["update"] == 2


def test_cleanup_expired_exports_mixed_progress_exits_cleanly(service, mock_db):
    """Cleanup should preserve successful updates and still terminate with mixed outcomes."""
    select_batches = [
        [
            {"job_id": "job-ok", "output_path": None},
            {"job_id": "job-fail", "output_path": None},
        ],
        [
            {"job_id": "job-fail", "output_path": None},
        ],
    ]
    update_attempts: list[str] = []

    def _execute_query(sql, params=None, commit=False):
        if sql.startswith("SELECT * FROM export_jobs"):
            if select_batches:
                return select_batches.pop(0)
            return []
        if sql.startswith("UPDATE export_jobs"):
            job_id = params[1]
            update_attempts.append(job_id)
            if job_id == "job-fail":
                raise RuntimeError("update failed")
            return None
        return []

    mock_db.execute_query.side_effect = _execute_query

    deleted = service.cleanup_expired_exports(batch_size=2)

    assert deleted == 0
    assert update_attempts.count("job-ok") == 1
    assert update_attempts.count("job-fail") == 2


def test_clean_old_exports(service, mock_db):
    """Test cleaning old export files."""
    export_dir = service.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    file_one = export_dir / "old1.chatbook"
    file_two = export_dir / "old2.chatbook"
    file_one.write_text("old")
    file_two.write_text("old")

    mock_db.execute_query.return_value = [
        ("old1", str(file_one)),
        ("old2", str(file_two)),
    ]

    count = service.clean_old_exports(days_old=7)

    assert count == 2
    assert not file_one.exists()
    assert not file_two.exists()
