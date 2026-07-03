"""Property-based tests for Chunker.chunk_text (audit F10)."""
import unicodedata

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


# --- Property adjustment (audit F10; see task-9-report.md for full writeup) ---
#
# `Chunker._sanitize_input` (app/core/Chunking/chunker.py:1355-1445) is a
# documented, intentional security-sanitization step that chunk_text always
# runs before chunking:
#
#   1. chunker.py:1428-1437 -- bidirectional-override control characters
#      (U+202A-E, U+2066-9) are *unconditionally* replaced with a space
#      ("could be used for spoofing"), regardless of test mode. Unlike null
#      bytes / other control chars (which the same method explicitly
#      preserves under PYTEST_CURRENT_TEST for property-testing purposes,
#      chunker.py:1386/1423), bidi-override chars have no such carve-out --
#      the omission is deliberate. Replacing one with a space can split a
#      single "word" (as tokenized by str.split() on the raw input) into two
#      tokens, so naive substring-preservation does not hold for text
#      containing them. We exclude these 9 codepoints from the fuzzed
#      alphabet: they change tokenization, not just content, which is a
#      different property than the one under test here.
#   2. chunker.py:1391-1394 -- NFC unicode normalization is applied
#      "to prevent various unicode-based attacks", but only when it does not
#      change the character count (so chunk offsets are preserved). This
#      guard does not imply the character *identity* is preserved: some
#      codepoints (e.g. CJK Compatibility Ideographs with a canonical
#      singleton decomposition, such as U+F900 -> U+8C48) normalize to a
#      different, length-preserving codepoint. Confirmed via manual repro:
#      `sanitize_input("豈")` returns "豈" (both render as "豈").
#      This is the documented, intended effect of canonicalizing
#      look-alike characters for security, not a bug -- so the expectation
#      mirrors that same NFC-if-length-preserving rule instead of asserting
#      raw byte-for-byte identity.
_BIDI_OVERRIDE_CHARS = "‪‫‬‭‮⁦⁧⁨⁩"


def _expected_word(word: str) -> str:
    """Mirror the length-preserving NFC normalization documented at
    chunker.py:1391-1394."""
    normalized = unicodedata.normalize("NFC", word)
    return normalized if len(normalized) == len(word) else word


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(exclude_characters=_BIDI_OVERRIDE_CHARS),
        min_size=1,
        max_size=5_000,
    ).filter(lambda s: s.strip())
)
def test_nonempty_input_content_is_preserved_in_chunks(text):
    chunks = Chunker().chunk_text(text, method="words", max_size=64, overlap=0)
    joined = " ".join(chunks)
    for word in text.split()[:5]:
        assert _expected_word(word) in joined
