import os
import types
from pathlib import Path

import pytest


# Target functions under test
from tldw_Server_API.app.core.Setup.install_manager import (
    _download_hf_file,
    _download_hf_dir,
)


@pytest.fixture()
def fake_hf_module(tmp_path, monkeypatch):
    """Provide a fake huggingface_hub module to avoid network calls.

    - hf_hub_download returns a path to a temp file containing known bytes.
    - snapshot_download populates the provided local_dir with a 'voices/' tree
      with a single file, then returns local_dir.
    """
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def hf_hub_download(*, repo_id: str, filename: str, revision: str | None = None, force_download: bool = False):  # noqa: ARG001
        # Create a deterministic source file in the fake cache
        src = cache_dir / Path(filename).name
        src.write_bytes(b"FAKE_MODEL_CONTENT")
        return str(src)

    def snapshot_download(
        *,
        repo_id: str,
        local_dir: str,
        revision: str | None = None,
        allow_patterns=None,
        force_download: bool = False,
    ):  # noqa: ARG001
        # Populate the requested local_dir with a 'voices' subtree and one file
        root = Path(local_dir)
        voices = root / "voices"
        voices.mkdir(parents=True, exist_ok=True)
        (voices / "voice-a.json").write_text("{\n  \"name\": \"A\"\n}\n", encoding="utf-8")
        return str(root)

    mod = types.SimpleNamespace(
        hf_hub_download=hf_hub_download,
        snapshot_download=snapshot_download,
    )
    # Ensure any "from huggingface_hub import ..." resolves to our fake module
    monkeypatch.setitem(os.sys.modules, "huggingface_hub", mod)
    return mod


def test_download_hf_file_skips_when_exists_without_force(tmp_path, monkeypatch, fake_hf_module):  # noqa: ARG001
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"OLD")

    # Track if hf_hub_download gets called (it should not)
    called = {"value": False}

    def guard_hf_hub_download(**kwargs):  # noqa: ANN001
        called["value"] = True
        return fake_hf_module.hf_hub_download(**kwargs)

    os.environ.pop("TLDW_SETUP_FORCE_DOWNLOADS", None)
    os.sys.modules["huggingface_hub"].hf_hub_download = guard_hf_hub_download  # type: ignore[attr-defined]

    _download_hf_file("repo/id", "onnx/model.onnx", dest)

    assert dest.read_bytes() == b"OLD"
    assert called["value"] is False


def test_download_hf_file_overwrites_with_force(tmp_path, monkeypatch, fake_hf_module):  # noqa: ARG001
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"OLD")

    os.environ["TLDW_SETUP_FORCE_DOWNLOADS"] = "1"
    try:
        _download_hf_file("repo/id", "onnx/model.onnx", dest)
    finally:
        os.environ.pop("TLDW_SETUP_FORCE_DOWNLOADS", None)

    assert dest.read_bytes() == b"FAKE_MODEL_CONTENT"


def test_download_hf_dir_skips_existing_without_force(tmp_path, monkeypatch, fake_hf_module):  # noqa: ARG001
    dest = tmp_path / "voices"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "preexisting.txt"
    marker.write_text("keep", encoding="utf-8")

    # Capture calls to snapshot_download (should not be called when skipping)
    called = {"value": False}

    def guard_snapshot_download(**kwargs):  # noqa: ANN001
        called["value"] = True
        return fake_hf_module.snapshot_download(**kwargs)

    os.environ.pop("TLDW_SETUP_FORCE_DOWNLOADS", None)
    os.sys.modules["huggingface_hub"].snapshot_download = guard_snapshot_download  # type: ignore[attr-defined]

    _download_hf_dir("repo/id", "voices", dest)

    assert marker.exists(), "existing dir should not be overwritten without --force"
    assert called["value"] is False


def test_download_hf_dir_copies_subtree_with_force(tmp_path, fake_hf_module):  # noqa: ARG001
    dest = tmp_path / "voices"
    # Put conflicting content in destination to verify overwrite
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "old.txt").write_text("old", encoding="utf-8")

    os.environ["TLDW_SETUP_FORCE_DOWNLOADS"] = "1"
    try:
        _download_hf_dir("repo/id", "voices", dest)
    finally:
        os.environ.pop("TLDW_SETUP_FORCE_DOWNLOADS", None)

    assert (dest / "voice-a.json").exists(), "expected file from mocked snapshot"
    assert not (dest / "old.txt").exists(), "destination should be cleaned when forcing"


def test_download_hf_dir_raises_if_subdir_missing(tmp_path, monkeypatch):


    """If the requested subdir is not present in snapshot, raise FileNotFoundError."""
    def empty_snapshot_download(
        *,
        repo_id: str,
        local_dir: str,
        revision: str | None = None,
        allow_patterns=None,
        force_download: bool = False,
    ):  # noqa: ARG001
        # Do not create the expected subdir
        return str(local_dir)

    mod = types.SimpleNamespace(snapshot_download=empty_snapshot_download)
    monkeypatch.setitem(os.sys.modules, "huggingface_hub", mod)

    dest = tmp_path / "voices"
    with pytest.raises(FileNotFoundError):
        _download_hf_dir("repo/id", "voices", dest)
