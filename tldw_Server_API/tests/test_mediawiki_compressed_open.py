"""
Tests for transparent compressed dump opening in MediaWiki importer.
Verifies that .xml, .xml.gz, and .xml.bz2 are read as UTF-8 text.
"""

import bz2
import gzip
from pathlib import Path

import pytest
from typing import Optional

from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki.Media_Wiki import (
    _open_dump_file_text,
)


@pytest.mark.unit
@pytest.mark.parametrize("suffix, writer", [
    (".xml", None),
    (".xml.gz", "gz"),
    (".xml.bz2", "bz2"),
])
def test_open_dump_file_text_supports_compression(tmp_path: Path, suffix: str, writer: Optional[str]):
    content = "<mediawiki>\n  <page><title>Test</title></page>\n</mediawiki>\n"
    dump_path = tmp_path / f"sample{suffix}"

    # Write with appropriate compressor (or plain text)
    if writer == "gz":
        with gzip.open(dump_path, mode="wt", encoding="utf-8") as f:
            f.write(content)
    elif writer == "bz2":
        with bz2.open(dump_path, mode="wt", encoding="utf-8") as f:
            f.write(content)
    else:
        dump_path.write_text(content, encoding="utf-8")

    # Read via importer helper
    with _open_dump_file_text(dump_path) as fh:
        read_back = fh.read()

    assert read_back == content
