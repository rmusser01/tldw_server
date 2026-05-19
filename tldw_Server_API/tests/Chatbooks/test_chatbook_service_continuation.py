import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.tests.Chatbooks.test_chatbook_service import mock_db, service  # noqa: F401


def _run_id(value: int) -> str:
    return "-".join(("run", str(value)))


def _cursor_id(value: int) -> str:
    return "-".join(("cursor", str(value)))


@pytest.mark.asyncio
async def test_continue_export_produces_linked_chatbook(service):
    """Continuation export should produce a chatbook linked to the original."""
    mock_evals_db = MagicMock()
    mock_evals_db.list_runs.return_value = (  # nosec B105
        [{"id": _run_id(5), "eval_id": "eval_1", "status": "completed"}],
        False,
    )
    service._evaluations_db = mock_evals_db

    success, message, path = await service.continue_chatbook_export(
        export_id="original-123",
        continuations=[
            {
                "evaluation_id": "eval_1",
                "continuation_token": _cursor_id(4),
            }
        ],
    )

    assert success is True
    assert message == "Continuation chatbook created successfully"
    assert path is not None
    assert Path(path).exists()

    with zipfile.ZipFile(path, "r") as zf:
        assert "manifest.json" in zf.namelist()
        manifest_data = json.loads(zf.read("manifest.json"))
        assert manifest_data["export_id"] == "original-123_cont_1"
        metadata = manifest_data.get("metadata", {})
        assert metadata.get("continues_export_id") == "original-123"

    mock_evals_db.list_runs.assert_called_once_with(
        eval_id="eval_1",
        limit=200,
        after=_cursor_id(4),
        return_has_more=True,
    )


@pytest.mark.asyncio
async def test_continue_export_with_more_data(service):
    """If there are still more rows, new continuation tokens should be produced."""
    mock_evals_db = MagicMock()
    mock_evals_db.list_runs.return_value = (  # nosec B105
        [
            {"id": _run_id(5), "eval_id": "eval_1", "status": "completed"},
            {"id": _run_id(6), "eval_id": "eval_1", "status": "completed"},
        ],
        True,
    )
    service._evaluations_db = mock_evals_db

    success, _message, path = await service.continue_chatbook_export(
        export_id="orig-456",
        continuations=[
            {
                "evaluation_id": "eval_1",
                "continuation_token": _cursor_id(4),
            }
        ],
    )

    assert success is True
    assert path is not None
    with zipfile.ZipFile(path, "r") as zf:
        manifest_data = json.loads(zf.read("manifest.json"))
        trunc = manifest_data.get("truncation", {})
        assert "evaluations" in trunc
        assert trunc["evaluations"]["truncated"] is True
        conts = trunc["evaluations"].get("continuations", [])
        assert len(conts) == 1
        assert conts[0]["continuation_token"] == _run_id(6)  # nosec B105


@pytest.mark.asyncio
async def test_continue_export_async_not_supported(service):
    """Async mode for continuation should return an error."""
    success, message, path = await service.continue_chatbook_export(
        export_id="x",
        continuations=[{"evaluation_id": "e1", "continuation_token": _cursor_id(1)}],  # nosec B105
        async_mode=True,
    )

    assert success is False
    assert message == "Async continuation exports are not yet supported"
    assert path is None


@pytest.mark.asyncio
async def test_continuation_of_continuation_uses_base_id(service):
    """Continuing a continuation should produce base_cont_2, not base_cont_1_cont_2."""
    mock_evals_db = MagicMock()
    mock_evals_db.list_runs.return_value = (  # nosec B105
        [{"id": _run_id(10), "eval_id": "eval_1", "status": "completed"}],
        False,
    )
    service._evaluations_db = mock_evals_db

    success, _message, path = await service.continue_chatbook_export(
        export_id="base_cont_1",
        continuations=[
            {
                "evaluation_id": "eval_1",
                "continuation_token": _cursor_id(9),
            }
        ],
    )

    assert success is True
    assert path is not None
    with zipfile.ZipFile(path, "r") as zf:
        manifest_data = json.loads(zf.read("manifest.json"))
        assert manifest_data["export_id"] == "base_cont_2"


@pytest.mark.asyncio
async def test_first_continuation_uses_base_cont_1(service):
    """First continuation of a base export should produce base_cont_1."""
    mock_evals_db = MagicMock()
    mock_evals_db.list_runs.return_value = (  # nosec B105
        [{"id": _run_id(5), "eval_id": "eval_1", "status": "completed"}],
        False,
    )
    service._evaluations_db = mock_evals_db

    success, _message, path = await service.continue_chatbook_export(
        export_id="my-export",
        continuations=[
            {
                "evaluation_id": "eval_1",
                "continuation_token": _cursor_id(4),
            }
        ],
    )

    assert success is True
    assert path is not None
    with zipfile.ZipFile(path, "r") as zf:
        manifest_data = json.loads(zf.read("manifest.json"))
        assert manifest_data["export_id"] == "my-export_cont_1"
