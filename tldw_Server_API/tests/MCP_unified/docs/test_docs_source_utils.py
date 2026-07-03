from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs import source_utils

pytestmark = pytest.mark.unit


def test_path_from_file_uri_uses_os_specific_path_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_url2pathname(path: str) -> str:
        calls.append(path)
        return r"C:\Users\Alice\My Docs\guide.md"

    monkeypatch.setattr(source_utils, "url2pathname", fake_url2pathname)

    path = source_utils.path_from_file_uri("file:///C:/Users/Alice/My%20Docs/guide.md")

    assert path == Path(r"C:\Users\Alice\My Docs\guide.md")  # nosec B101
    assert calls == ["/C:/Users/Alice/My%20Docs/guide.md"]  # nosec B101
