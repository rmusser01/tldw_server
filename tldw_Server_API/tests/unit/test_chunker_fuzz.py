"""Property-based tests for Chunker.chunk_text (audit F10)."""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.core.Chunking.chunker import Chunker


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(
    text=st.text(max_size=5_000),
    max_size=st.integers(min_value=16, max_value=1_024),
    overlap=st.integers(min_value=0, max_value=15),
)
def test_chunk_text_returns_list_of_strings(text, max_size, overlap):
    chunks = Chunker().chunk_text(text, method="words", max_size=max_size, overlap=overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(max_size=st.integers(min_value=16, max_value=1_024))
def test_empty_input_yields_no_chunks(max_size):
    assert Chunker().chunk_text("", method="words", max_size=max_size, overlap=0) == []


# --- Property fix (audit F10 review finding; see task-9-report.md) ---
#
# `Chunker._sanitize_input` (app/core/Chunking/chunker.py:1355-1445) is a
# documented, intentional security-sanitization step that chunk_text always
# runs before chunking. Two of its transforms are relevant here:
#
#   1. chunker.py:1391-1394 -- NFC unicode normalization is applied only when
#      it does not change the *whole string's* character count (so chunk
#      offsets are preserved).
#   2. chunker.py:1428-1437 -- bidirectional-override control characters
#      (U+202A-E, U+2066-9) are unconditionally replaced with a space
#      ("could be used for spoofing"), which can split a raw "word" into two
#      tokens.
#
# The previous version of this test reimplemented rule (1) on a per-word
# basis (normalizing each word individually and keeping the normalized form
# only if that single word's NFC form was length-preserving). That diverges
# from the sanitizer's real whole-string gate: a text whose overall NFC
# normalization changes length is left entirely un-normalized by the
# sanitizer, even though some individual word within it might normalize
# losslessly in isolation. That mismatch produced a false-positive failure
# for correct sanitizer behavior -- repro:
# `Chunker().chunk_text("é 敖", method="words", max_size=64,
# overlap=0)` leaves the text un-normalized (the whole-string NFC form has a
# different length), but the old per-word helper predicted normalization of
# the second word regardless.
#
# Fix: stop re-deriving the gating logic in the test and instead call
# `Chunker()._sanitize_input(text)` once on the whole input, then assert that
# every expected word came from the real sanitizer's output. This also makes
# the bidi-override alphabet exclusion unnecessary: since expected words are
# now split from the *sanitized* text (not the raw text), a bidi char being
# replaced with a space is already reflected in the expected tokenization.
@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(text=st.text(min_size=1, max_size=5_000).filter(lambda s: s.strip()))
def test_nonempty_input_content_is_preserved_in_chunks(text):
    chunker = Chunker()
    # assumes _sanitize_input is a pure function of text (same call chunk_text makes internally)
    sanitized = chunker._sanitize_input(text)
    if not sanitized.strip():
        # Sanitization (e.g. bidi-override removal) can reduce an input that
        # was non-blank pre-sanitization down to whitespace-only; nothing to
        # check in that case.
        return
    chunks = chunker.chunk_text(text, method="words", max_size=64, overlap=0)
    joined = " ".join(chunks)
    for word in sanitized.split()[:5]:
        assert word in joined
