from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as db_deps


pytestmark = pytest.mark.unit


def test_get_db_path_for_user_testing_uses_configured_base_without_mutating_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)
    monkeypatch.setenv("TESTING", "y")
    monkeypatch.setenv("TLDW_TEST_RUN_ID", "truthy-y")
    monkeypatch.setitem(db_deps.settings, "USER_DB_BASE_DIR", tmp_path)

    resolved = db_deps._get_db_path_for_user(7)

    assert resolved == tmp_path / "7" / "Media_DB_v2.db"
    assert "USER_DB_BASE_DIR" not in os.environ
