import zipfile
from pathlib import Path

import pytest


@pytest.mark.unit
def test_process_zip_of_epubs_rejects_absolute_sibling_prefix_escape(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Books import (
        Book_Processing_Lib as book_lib,
    )

    extract_root = tmp_path / "epub_extract"
    sibling_escape = tmp_path / "epub_extract_evil"

    class _FixedTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.name = str(extract_root)

        def __enter__(self):
            extract_root.mkdir(parents=True, exist_ok=True)
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(book_lib.tempfile, "TemporaryDirectory", _FixedTemporaryDirectory)

    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(str(sibling_escape / "payload.epub"), b"not a real epub")

    results = book_lib.process_zip_of_epubs(str(archive_path))

    assert results and results[0]["status"] == "Error"
    assert not (sibling_escape / "payload.epub").exists()
