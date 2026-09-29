"""PST metadata regressions using the native parser's public API shape."""

import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as lib


@pytest.fixture
def process_native_message(monkeypatch):
    """Substitute only the optional libpff boundary; run the real email conversion."""
    def process(**fields):
        message = SimpleNamespace(subject="Synthetic PST", plain_text_body=b"Synthetic body", **fields)
        folder = SimpleNamespace(number_of_sub_folders=0, number_of_sub_messages=1,
                                 get_sub_message=lambda _index: message)
        pst = SimpleNamespace(open=lambda _path: None, close=lambda: None,
                              get_root_folder=lambda: folder)
        monkeypatch.setitem(sys.modules, "pypff", SimpleNamespace(file=lambda: pst))
        result = lib.process_pst_bytes(file_bytes=b"synthetic boundary", pst_name="synthetic.pst",
                                       perform_chunking=False)
        assert result[0]["status"] == "Success"
        return result[0]["metadata"]["email"]
    return process


def test_pst_native_datetime_is_preserved_without_transport_headers(process_native_message):
    metadata = process_native_message(delivery_time=datetime(2020, 1, 2, 3, 4, 5))
    assert metadata["date"] == "Thu, 02 Jan 2020 03:04:05 +0000"


@pytest.mark.parametrize("field,expected", [
    ("from", "alice@example.com"),
    ("to", "bob@example.com"),
    ("cc", "carol@example.com"),
    ("bcc", "dave@example.com"),
    ("date", "Thu, 02 Jan 2020 03:04:05 +0000"),
    ("message_id", "<synthetic-pst@example.com>"),
])
def test_pst_transport_headers_supply_missing_native_metadata(process_native_message, field, expected):
    headers = (
        "From: Alice <alice@example.com>\r\n"
        "To: Bob <bob@example.com>\r\n"
        "Cc: Carol <carol@example.com>\r\n"
        "Bcc: Dave <dave@example.com>\r\n"
        "Date: Thu, 02 Jan 2020 03:04:05 +0000\r\n"
        "Message-ID: <synthetic-pst@example.com>\r\n"
        "Content-Type: application/ms-tnef\r\n\r\n"
    )
    metadata = process_native_message(transport_headers=headers)
    assert metadata[field] == expected


def test_pst_native_date_precedes_transport_header_date(process_native_message):
    metadata = process_native_message(delivery_time=datetime(2021, 1, 2, 3, 4, 5),
        transport_headers="Date: Thu, 02 Jan 2020 03:04:05 +0000\r\n\r\n")
    assert metadata["date"] == "Sat, 02 Jan 2021 03:04:05 +0000"
