import os

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig, ChunkingMethod
from tldw_Server_API.app.core.Chunking.chunker import Chunker


@pytest.fixture(autouse=True)
def testing_env():
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)


def words_to_text(words):

    return " ".join(words)


@hyp_settings(deadline=None)
@given(
    total_words=st.integers(min_value=10, max_value=200),
    max_size=st.integers(min_value=3, max_value=40),
    overlap=st.integers(min_value=0, max_value=10),
)
def test_words_overlap_property(total_words, max_size, overlap):
    # Constrain overlap < max_size
    if overlap >= max_size:
        overlap = max_size - 1
        if overlap < 0:
            overlap = 0

    # Build a simple word list w0 w1 ...
    words = [f"w{i}" for i in range(total_words)]
    text = words_to_text(words)

    cfg = ChunkerConfig(
        default_method=ChunkingMethod.WORDS, default_max_size=max_size, default_overlap=overlap, language="en"
    )
    ck = Chunker(config=cfg)
    chunks = ck.chunk_text(text, method=ChunkingMethod.WORDS.value, max_size=max_size, overlap=overlap)

    if len(chunks) <= 1 or overlap == 0:
        # Nothing to assert about overlap
        return

    # Check that last <overlap> words of chunk i equals first <overlap> of chunk i+1
    for i in range(len(chunks) - 1):
        a = chunks[i].split()
        b = chunks[i + 1].split()
        if len(a) >= overlap and len(b) >= overlap:
            assert a[-overlap:] == b[:overlap]


# --------------------------------------------------------------------------- #
# Additional word-chunking invariants (RA4, plan Task 7): reconstruction, the
# chunk-size bound, and coverage. These complement the overlap-adjacency check
# above; an off-by-one in the overlap stride breaks reconstruction.
# --------------------------------------------------------------------------- #
def _make_chunks(total_words, max_size, overlap):
    if overlap >= max_size:
        overlap = max(0, max_size - 1)
    words = [f"w{i}" for i in range(total_words)]
    cfg = ChunkerConfig(
        default_method=ChunkingMethod.WORDS,
        default_max_size=max_size,
        default_overlap=overlap,
        language="en",
    )
    ck = Chunker(config=cfg)
    chunks = ck.chunk_text(
        " ".join(words), method=ChunkingMethod.WORDS.value, max_size=max_size, overlap=overlap
    )
    return words, chunks, overlap


@hyp_settings(deadline=None, max_examples=150)
@given(
    total_words=st.integers(min_value=1, max_value=200),
    max_size=st.integers(min_value=1, max_value=40),
    overlap=st.integers(min_value=0, max_value=10),
)
def test_words_reconstruction_property(total_words, max_size, overlap):
    """Streaming the chunk words in order and keeping each word's first
    occurrence must reconstruct the original sequence — no word is dropped or
    reordered. (A boundary chunk may legitimately repeat a word at very small
    max_size, so exact non-dup reconstruction is not asserted; the ordered-
    first-occurrence view still fails on any off-by-one that skips/reorders.)"""
    words, chunks, _overlap = _make_chunks(total_words, max_size, overlap)
    assert chunks, "non-empty input produced no chunks"

    seen: list[str] = []
    seen_set: set[str] = set()
    for chunk in chunks:
        for word in chunk.split():
            if word not in seen_set:
                seen.append(word)
                seen_set.add(word)
    assert seen == words, "ordered chunk words do not reconstruct the source sequence"


@hyp_settings(deadline=None, max_examples=150)
@given(
    total_words=st.integers(min_value=1, max_value=200),
    max_size=st.integers(min_value=1, max_value=40),
    overlap=st.integers(min_value=0, max_value=10),
)
def test_words_chunk_size_bound(total_words, max_size, overlap):
    """No chunk exceeds max_size words, and every source word appears somewhere."""
    words, chunks, _overlap = _make_chunks(total_words, max_size, overlap)
    for chunk in chunks:
        assert len(chunk.split()) <= max_size, "a chunk exceeded max_size words"
    covered = set()
    for chunk in chunks:
        covered.update(chunk.split())
    assert covered == set(words), "chunks did not cover every source word"
