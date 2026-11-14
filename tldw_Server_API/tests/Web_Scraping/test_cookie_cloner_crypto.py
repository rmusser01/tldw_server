"""Unit tests for cookie_cloner crypto and parsers.

These tests use synthetic AES-CBC vectors and hand-built Safari binary
page blobs — no browser profile databases, no network, fully deterministic.
The module under test reads real browser cookie stores in its get_*_cookies
functions; those are intentionally NOT exercised here (environment-bound).
"""

from __future__ import annotations

import struct

import pytest
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad

from tldw_Server_API.app.core.Web_Scraping.cookie_scraping import cookie_cloner

pytestmark = pytest.mark.unit

# The module decrypts Chrome/Edge cookies with a fixed 16-space IV.
_FIXED_IV = b" " * 16
_KEY = b"\x01" * 16  # synthetic AES-128 key
_SECRET = b"session-secret-value"


def _encrypt(plaintext: bytes, key: bytes = _KEY, prefix: bytes = b"v10") -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, IV=_FIXED_IV)
    return prefix + cipher.encrypt(pad(plaintext, AES.block_size))


@pytest.mark.parametrize("prefix", [b"v10", b"v11", b""])
def test_decrypt_chrome_cookie_round_trips(prefix: bytes):
    decrypted = cookie_cloner.decrypt_chrome_cookie(_encrypt(_SECRET, prefix=prefix), _KEY)
    assert decrypted == _SECRET


@pytest.mark.parametrize("prefix", [b"v10", b"v11", b""])
def test_decrypt_edge_cookie_round_trips(prefix: bytes):
    decrypted = cookie_cloner.decrypt_edge_cookie(_encrypt(_SECRET, prefix=prefix), _KEY)
    assert decrypted == _SECRET


def test_decrypt_chrome_cookie_with_wrong_key_does_not_leak_plaintext():
    wrong_key = b"\x02" * 16
    decrypted = cookie_cloner.decrypt_chrome_cookie(_encrypt(_SECRET), wrong_key)
    assert decrypted != _SECRET
    assert _SECRET not in decrypted


def test_decrypt_edge_cookie_with_wrong_key_does_not_leak_plaintext():
    wrong_key = b"\x03" * 16
    decrypted = cookie_cloner.decrypt_edge_cookie(_encrypt(_SECRET), wrong_key)
    assert decrypted != _SECRET
    assert _SECRET not in decrypted


def test_decrypt_chrome_cookie_malformed_input_fails_without_plaintext():
    # Prefix-only input yields empty ciphertext; current behavior raises
    # IndexError on the padding byte. Characterized so a future hardening
    # change is deliberate.
    with pytest.raises(IndexError):
        cookie_cloner.decrypt_chrome_cookie(b"v10", _KEY)


def test_decrypt_chrome_cookie_unicode_round_trip():
    secret = "värde-秘密".encode("utf-8")
    decrypted = cookie_cloner.decrypt_chrome_cookie(_encrypt(secret), _KEY)
    assert decrypted == secret


def _safari_cookie_blob(domain: bytes, name: bytes, value: bytes) -> bytes:
    """Build a minimal Safari cookie record per parse_safari_cookie's layout.

    Layout (little-endian ints): [0:4] flags, [4:8] url_offset,
    [8:12] name_offset, [12:16] value offset field (ignored by parser),
    [16:20] value_offset, then NUL-terminated domain/name/value strings.
    """
    header = bytearray(20)
    strings = domain + b"\x00" + name + b"\x00" + value + b"\x00"
    struct.pack_into("<i", header, 4, 20)   # url/domain offset
    struct.pack_into("<i", header, 8, 20 + len(domain) + 1)   # name offset
    struct.pack_into("<i", header, 12, 0)   # ignored by parser
    struct.pack_into("<i", header, 16, 20 + len(domain) + 1 + len(name) + 1)  # value offset
    return bytes(header) + strings


def _safari_page(blobs: list[bytes]) -> bytes:
    page = bytearray(8 + 4 * len(blobs))
    struct.pack_into(">i", page, 4, len(blobs))  # big-endian cookie count
    offset = len(page)
    offsets = []
    for blob in blobs:
        offsets.append(offset)
        offset += len(blob)
    for i, off in enumerate(offsets):
        struct.pack_into(">i", page, 8 + i * 4, off)
    return bytes(page) + b"".join(blobs)


def test_parse_safari_page_extracts_matching_domain_cookie():
    page = _safari_page([
        _safari_cookie_blob(b".example.com", b"session", b"abc123"),
        _safari_cookie_blob(b".other.net", b"unrelated", b"nope"),
    ])
    cookies = cookie_cloner.parse_safari_page(page, "example.com")
    assert cookies == {"session": "abc123"}


def test_parse_safari_cookie_ignores_non_matching_domain():
    blob = _safari_cookie_blob(b".other.net", b"unrelated", b"nope")
    assert cookie_cloner.parse_safari_cookie(blob, "example.com") is None


def test_parse_safari_cookie_malformed_blob_returns_none():
    # Truncated garbage must fail soft (None), never raise out of the parser.
    assert cookie_cloner.parse_safari_cookie(b"\x00" * 8, "example.com") is None
    assert cookie_cloner.parse_safari_cookie(b"", "example.com") is None
