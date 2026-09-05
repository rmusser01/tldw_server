import hashlib
import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_compatibility import (
    UnstableFingerprintError,
    build_fingerprint,
    canonical_sha256,
    compare_fingerprints,
    hash_file_stable,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import Fingerprint


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
