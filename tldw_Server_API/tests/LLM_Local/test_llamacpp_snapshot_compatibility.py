"""Verify snapshot content identity, race rejection, and bounded hash reuse."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.core.Local_LLM import llamacpp_snapshot_compatibility as compatibility
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_compatibility import (
    UnstableFingerprintError,
    build_fingerprint,
    canonical_sha256,
    compare_fingerprints,
    hash_file_stable,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import Fingerprint

pytestmark = pytest.mark.unit


@pytest.fixture
def read_bytes(monkeypatch):
    """Count actual streamed bytes while preserving real filesystem reads."""
    counts = []
    original_read = os.read

    def counted_read(fd: int, size: int) -> bytes:
        chunk = original_read(fd, size)
        counts.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", counted_read)
    return counts


def test_cached_polling_streams_unchanged_files_only_once(tmp_path: Path, read_bytes):
    cache = compatibility.FingerprintHashCache()
    model = tmp_path / "model.gguf"
    executable = tmp_path / "server"
    projector = tmp_path / "projector.gguf"
    model.write_bytes(b"model")
    executable.write_bytes(b"server")
    projector.write_bytes(b"projector")

    results = [
        build_fingerprint(
            model=model,
            executable=executable,
            projector=projector,
            effective_options={},
            adapters=[],
            cache=cache,
        )
        for _ in range(3)
    ]

    assert results[0] == results[1] == results[2]
    assert sum(read_bytes) == len(b"modelserverprojector")


@pytest.mark.parametrize("replacement", [False, True])
def test_cache_invalidates_same_size_change_with_restored_mtime(
    tmp_path: Path,
    read_bytes,
    replacement: bool,
):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"before")
    initial = source.stat()
    cache.hash_file(source)
    target = tmp_path / "replacement" if replacement else source
    target.write_bytes(b"after!")
    os.utime(target, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    if replacement:
        target.replace(source)

    assert cache.hash_file(source) == hashlib.sha256(b"after!").hexdigest()
    assert sum(read_bytes) == 12


def test_unstable_hash_is_not_cached(tmp_path: Path, monkeypatch, read_bytes):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"before")
    original_read = os.read
    changed = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(fd, size)
        if chunk and not changed:
            source.write_bytes(b"after!")
            changed = True
        return chunk

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(UnstableFingerprintError):
        cache.hash_file(source)
    assert cache.hash_file(source) == hashlib.sha256(b"after!").hexdigest()
    assert sum(read_bytes) == 12


@pytest.mark.parametrize("replace_on_stat", [1, 2])
def test_cache_hit_rejects_path_replacement_during_validation(
    tmp_path: Path,
    monkeypatch,
    replace_on_stat: int,
):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"before")
    cache.hash_file(source)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"after!")
    original_fstat = os.fstat
    stat_calls = 0

    def replace_after_stat(fd: int):
        nonlocal stat_calls
        result = original_fstat(fd)
        stat_calls += 1
        if stat_calls == replace_on_stat:
            replacement.replace(source)
        return result

    monkeypatch.setattr(os, "fstat", replace_after_stat)
    with pytest.raises(UnstableFingerprintError):
        cache.hash_file(source)
    assert cache.hash_file(source) == hashlib.sha256(b"after!").hexdigest()


def test_cached_file_replaced_by_symlink_is_rejected(tmp_path: Path):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"before")
    cache.hash_file(source)
    target = tmp_path / "target"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(OSError):
        cache.hash_file(source)


def test_cache_evicts_least_recently_used_file(tmp_path: Path, read_bytes):
    cache = compatibility.FingerprintHashCache(max_entries=2)
    sources = [tmp_path / name for name in ("a", "b", "c")]
    for source in sources:
        source.write_bytes(source.name.encode())
    for index in (0, 1, 0, 2, 0):
        cache.hash_file(sources[index])
    assert sum(read_bytes) == 3
    cache.hash_file(sources[1])
    assert sum(read_bytes) == 4


def test_concurrent_cache_lookups_stream_file_once(tmp_path: Path, read_bytes):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    content = b"model" * 300_000
    source.write_bytes(content)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cache.hash_file, [source] * 8))
    assert results == [hashlib.sha256(content).hexdigest()] * 8
    assert sum(read_bytes) == len(content)


def test_failed_read_is_not_cached(tmp_path: Path, monkeypatch, read_bytes):
    cache = compatibility.FingerprintHashCache()
    source = tmp_path / "model.gguf"
    source.write_bytes(b"model")

    def fail_read(fd: int, size: int) -> bytes:
        raise OSError("injected read failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", fail_read)
        with pytest.raises(OSError, match="injected read failure"):
            cache.hash_file(source)
    assert cache.hash_file(source) == hashlib.sha256(b"model").hexdigest()
    assert sum(read_bytes) == 5


def test_unknown_and_changed_model_never_match():
    saved = Fingerprint(
        model_sha256="a" * 64,
        executable_sha256="b" * 64,
        effective_options_sha256="c" * 64,
        adapters_sha256="d" * 64,
    )

    assert compare_fingerprints(saved, None) == ["compatibility_unknown"]
    changed = saved.model_copy(update={"model_sha256": "e" * 64})
    assert compare_fingerprints(saved, changed) == ["model_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_sha256", "e" * 64),
        ("executable_sha256", "e" * 64),
        ("projector_sha256", "e" * 64),
        ("effective_options_sha256", "e" * 64),
        ("adapters_sha256", "e" * 64),
    ],
)
def test_each_fingerprint_mismatch_is_reported(field: str, value: str):
    saved = Fingerprint(
        model_sha256="a" * 64,
        executable_sha256="b" * 64,
        effective_options_sha256="c" * 64,
        adapters_sha256="d" * 64,
    )

    assert compare_fingerprints(saved, saved.model_copy(update={field: value})) == [field]


def test_fingerprint_uses_contents_and_canonical_configuration(tmp_path: Path):
    model = tmp_path / "renamed-model.gguf"
    executable = tmp_path / "llama-server"
    projector = tmp_path / "projector.gguf"
    model.write_bytes(b"model")
    executable.write_bytes(b"executable")
    projector.write_bytes(b"projector")

    result = build_fingerprint(
        model=model,
        executable=executable,
        projector=projector,
        effective_options={"threads": 4, "ctx": 4096},
        adapters=[{"sha256": "f" * 64, "scale": 0.5}],
    )

    assert result.model_sha256 == hashlib.sha256(b"model").hexdigest()
    assert result.projector_sha256 == hashlib.sha256(b"projector").hexdigest()
    assert result.effective_options_sha256 == canonical_sha256({"ctx": 4096, "threads": 4})


def test_fingerprint_rejects_symlink_sources(tmp_path: Path):
    real = tmp_path / "real.gguf"
    real.write_bytes(b"model")
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(real)

    with pytest.raises(OSError):
        build_fingerprint(
            model=alias,
            executable=real,
            effective_options={},
            adapters=[],
        )


def test_fingerprint_rejects_atomic_path_replacement_during_read(tmp_path: Path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"original")
    replacement = tmp_path / "replacement.gguf"
    replacement.write_bytes(b"replaced")
    original_read = os.read
    replaced = False

    def replace_after_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(fd, size)
        if chunk and not replaced:
            replacement.replace(source)
            replaced = True
        return chunk

    monkeypatch.setattr(os, "read", replace_after_read)
    with pytest.raises(UnstableFingerprintError):
        hash_file_stable(source)


@given(
    st.one_of(
        st.text(alphabet="0123456789abcdef", max_size=63),
        st.text(alphabet="0123456789abcdef", min_size=65, max_size=80),
        st.just("A" * 64),
        st.just("z" * 64),
    )
)
def test_fingerprint_rejects_non_sha256_digest_values(value: str):
    with pytest.raises(ValidationError):
        Fingerprint(
            model_sha256=value,
            executable_sha256="b" * 64,
            effective_options_sha256="c" * 64,
            adapters_sha256="d" * 64,
        )
