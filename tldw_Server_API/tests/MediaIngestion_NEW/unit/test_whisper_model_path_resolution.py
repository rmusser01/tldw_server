"""Managed Whisper resolution must not give outside CWD paths precedence."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

pytestmark = pytest.mark.unit


@pytest.fixture
def model_env(monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(atlib, "WHISPER_MODEL_BASE_DIR", root)
    monkeypatch.setattr(atlib.WhisperModel, "default_download_root", str(root))
    snapshot = root / "models--org--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    downloads, delegates = [], []

    def download(identifier, **kwargs):
        downloads.append((identifier, kwargs))
        return str(snapshot)

    def delegate(**kwargs):
        delegates.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=download))
    monkeypatch.setattr(atlib, "_get_original_whisper_model", lambda: delegate)
    return SimpleNamespace(root=root, cwd=cwd, snapshot=snapshot, downloads=downloads, delegates=delegates)


@pytest.mark.parametrize("identifier", ["tiny.en", "org/model"])
@pytest.mark.parametrize("shadow_exists", [False, True])
def test_remote_identifier_ignores_outside_cwd_directory(model_env, monkeypatch, identifier, shadow_exists):
    shadow = model_env.cwd / identifier
    if shadow_exists:
        shadow.mkdir(parents=True)
    original_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == shadow or path == Path(identifier):
            pytest.fail("probed the outside CWD model path")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    assert atlib._normalize_whisper_model_identifier(identifier) == identifier
    atlib.WhisperModel(identifier)
    assert model_env.downloads[0][0] == identifier
    assert model_env.delegates[0]["model_size_or_path"] == str(model_env.snapshot)


@pytest.mark.parametrize("form", ["absolute", "base-relative", "cwd-under-root", "standard-alias"])
def test_managed_local_models_bypass_download(model_env, monkeypatch, form):
    local = model_env.root / "local"
    local.mkdir()
    identifier = str(local)
    if form == "base-relative":
        identifier = "./local"
    elif form == "cwd-under-root":
        monkeypatch.chdir(model_env.root)
        identifier = "local"
    elif form == "standard-alias":
        identifier = "local"
    atlib.WhisperModel(identifier)
    assert model_env.downloads == []
    assert model_env.delegates[0]["model_size_or_path"] == str(local)


@pytest.mark.parametrize("identifier", ["../outside", "/outside/model"])
def test_explicit_escape_rejected_before_filesystem_probe(model_env, monkeypatch, identifier):
    original_stat = Path.stat
    outside = Path(identifier) if Path(identifier).is_absolute() else model_env.root.parent / "outside"

    def guarded_stat(path, *args, **kwargs):
        if path == outside or path == Path(identifier):
            pytest.fail("probed an explicit outside path")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ValueError, match="must resolve under"):
        atlib._normalize_whisper_model_identifier(identifier)


def test_remote_resolution_forwards_cache_offline_revision_and_token(model_env):
    atlib.WhisperModel("org/model", local_files_only=True, revision="pinned", use_auth_token="test-token")
    assert model_env.downloads == [("org/model", {
        "cache_dir": str(model_env.root), "local_files_only": True,
        "revision": "pinned", "use_auth_token": "test-token",
    })]
    assert model_env.delegates[0]["local_files_only"] is True


@pytest.mark.parametrize("kind", ["outside", "symlink", "missing"])
def test_downloaded_snapshot_must_be_existing_managed_directory(model_env, monkeypatch, kind):
    result = model_env.cwd / "outside"
    result.mkdir()
    if kind == "symlink":
        result = model_env.root / "linked"
        result.symlink_to(model_env.snapshot, target_is_directory=True)
    elif kind == "missing":
        result = model_env.root / "missing"
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=lambda *a, **k: str(result)))
    with pytest.raises(ValueError):
        atlib.WhisperModel("org/model")
    assert model_env.delegates == []


def test_snapshot_allows_huggingface_internal_artifact_links(model_env):
    blob = model_env.root / "blobs" / "hash"
    blob.parent.mkdir()
    blob.write_bytes(b"model")
    (model_env.snapshot / "model.bin").symlink_to(blob)
    atlib.WhisperModel("org/model")
    assert model_env.delegates[0]["model_size_or_path"] == str(model_env.snapshot)


@pytest.mark.parametrize("identifier", ["tiny.en", "org/model"])
def test_check_model_exists_does_not_download_or_follow_cache_directory_links(model_env, identifier):
    outside = model_env.cwd / "outside"
    outside.mkdir()
    link = model_env.root / identifier
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    assert atlib.check_model_exists(identifier) is False
    assert model_env.downloads == []


def test_delegate_type_error_does_not_repeat_download(model_env, monkeypatch):
    def delegate(**kwargs):
        raise TypeError("incompatible delegate")
    monkeypatch.setattr(atlib, "_get_original_whisper_model", lambda: delegate)
    with pytest.raises(RuntimeError, match="incompatible delegate"):
        atlib.WhisperModel("org/model")
    assert len(model_env.downloads) == 1


@pytest.mark.parametrize("identifier", ["tiny.en", "org/model", "org_model"])
def test_check_model_exists_recognizes_managed_models_without_network(model_env, identifier):
    (model_env.root / identifier).mkdir(parents=True)
    assert atlib.check_model_exists(identifier) is True
    assert model_env.downloads == []


def test_download_root_override_is_forwarded_to_cache(model_env):
    nested_root = model_env.root / "nested"
    nested_root.mkdir()
    snapshot = nested_root / "snapshot"
    snapshot.mkdir()
    sys.modules["faster_whisper.utils"].download_model = lambda *a, **k: (
        model_env.downloads.append(k) or str(snapshot)
    )
    atlib.WhisperModel("tiny", download_root=str(nested_root))
    assert model_env.downloads[0]["cache_dir"] == str(nested_root)
    assert model_env.delegates[0]["model_size_or_path"] == str(snapshot)


def test_linked_model_parent_is_rejected_before_probing_its_children(model_env, monkeypatch):
    outside = model_env.cwd / "outside"
    outside.mkdir()
    linked_parent = model_env.root / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    original_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == linked_parent / "child":
            pytest.fail("followed linked parent to probe its child")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ValueError, match="symlinks"):
        atlib._normalize_whisper_model_identifier(str(linked_parent / "child"))


def test_offline_cache_miss_never_delegates_to_loader(model_env, monkeypatch):
    def missing(identifier, **kwargs):
        assert kwargs["local_files_only"] is True
        raise ValueError("could not be found in local cache")
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=missing))
    with pytest.raises(ValueError, match="could not be loaded"):
        atlib.WhisperModel("tiny", local_files_only=True)
    assert model_env.delegates == []


@pytest.mark.parametrize("account_exists", [False, True])
def test_model_identifier_does_not_query_system_accounts(model_env, monkeypatch, account_exists):
    pwd = pytest.importorskip("pwd")
    lookups = []

    def lookup(account):
        lookups.append(account)
        if not account_exists:
            raise KeyError(account)
        return SimpleNamespace(pw_dir=str(model_env.cwd))

    monkeypatch.setattr(pwd, "getpwnam", lookup)
    with pytest.raises(ValueError):
        atlib.validate_whisper_model_identifier("~probe-account/model")
    assert lookups == []


def test_model_directory_with_literal_tilde_stays_under_managed_root(model_env):
    local = model_env.root / "~probe-account" / "model"
    local.mkdir(parents=True)
    assert atlib._normalize_whisper_model_identifier("~probe-account/model") == str(local)


@pytest.mark.parametrize("identifier", ["org/model", "Systran/faster-whisper-large-v3"])
def test_hub_identifier_never_reaches_absolute_path_existence_probe(model_env, monkeypatch, identifier):
    exists_calls = []
    original_exists = Path.exists

    def track_exists(path):
        exists_calls.append(path)
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", track_exists)
    assert atlib.check_model_exists(identifier) is False
    assert exists_calls == []
    assert model_env.downloads == []
