"""Checkpoint path guards precede filesystem resolution."""
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service.checkpoint import CheckpointManager


@pytest.mark.unit
@pytest.mark.parametrize("absolute", [False, True])
def test_checkpoint_rejects_outside_path_before_resolving(tmp_path, monkeypatch, absolute):
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    outside = tmp_path / "checkpoints-sibling" / "private.json"
    path_value = outside if absolute else Path("../checkpoints-sibling/private.json")
    original_resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        if path == outside or ".." in path.parts:
            pytest.fail("resolved an outside checkpoint path before containment")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        manager._resolve_checkpoint_path(path_value)


@pytest.mark.unit
def test_checkpoint_canonical_check_rejects_inside_link_to_outside(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    outside = tmp_path / "outside.json"
    outside.write_text("private", encoding="utf-8")
    (manager.checkpoint_dir / "link.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        manager._resolve_checkpoint_path("link.json")


@pytest.mark.unit
def test_checkpoint_accepts_directory_itself_and_managed_absolute_relative_paths(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    expected = manager.checkpoint_dir / "valid.json"
    assert manager._resolve_checkpoint_path(".") == manager.checkpoint_dir
    assert manager._resolve_checkpoint_path("valid.json") == expected
    assert manager._resolve_checkpoint_path(expected) == expected


@pytest.mark.unit
def test_checkpoint_case_normalization_does_not_change_opened_path(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service import checkpoint

    manager = CheckpointManager(checkpoint_dir=tmp_path / "MixedCaseDirectory")
    original = manager.checkpoint_dir / "MixedCaseFile.json"
    original.write_text("original case", encoding="utf-8")
    comparison_paths = SimpleNamespace(
        abspath=os.path.abspath, normcase=lambda value: str(value).lower(), join=os.path.join,
    )
    monkeypatch.setattr(checkpoint, "os", SimpleNamespace(path=comparison_paths))
    assert manager._resolve_checkpoint_path(original).read_text(encoding="utf-8") == "original case"


@pytest.mark.unit
def test_checkpoint_equal_root_still_checks_current_canonical_path(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    manager.checkpoint_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    manager.checkpoint_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        manager._resolve_checkpoint_path(".")


@pytest.mark.unit
@pytest.mark.parametrize("case", ["outside-absolute", "outside-relative", "canonical-link"])
def test_checkpoint_windows_case_sibling_never_passes_boundary(monkeypatch, case):
    import ntpath
    from pathlib import PureWindowsPath
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service import checkpoint

    resolutions = []

    class CanonicalWindowsPath(PureWindowsPath):
        def resolve(self, *, strict=False):
            resolutions.append(str(self))
            if str(self) == r"C:\Cache\link.json":
                return CanonicalWindowsPath(r"C:\cache\secret.json")
            return self

    manager = object.__new__(CheckpointManager)
    manager.checkpoint_dir = CanonicalWindowsPath(r"C:\Cache")
    monkeypatch.setattr(checkpoint, "Path", CanonicalWindowsPath)
    monkeypatch.setattr(checkpoint, "os", SimpleNamespace(path=ntpath))
    value = {
        "outside-absolute": r"C:\cache\secret.json",
        "outside-relative": r"..\cache\secret.json",
        "canonical-link": "link.json",
    }[case]
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        manager._resolve_checkpoint_path(value)
    assert resolutions == ([r"C:\Cache\link.json"] if case == "canonical-link" else [])


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("MixedCase.json", r"C:\Cache\MixedCase.json"),
    (r"C:\Cache\MixedCase.json", r"C:\Cache\MixedCase.json"),
    (".", r"C:\Cache"),
])
def test_checkpoint_preserves_canonical_windows_paths(monkeypatch, value, expected):
    import ntpath
    from pathlib import PureWindowsPath
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service import checkpoint

    class CanonicalWindowsPath(PureWindowsPath):
        def resolve(self, *, strict=False):
            return self

    manager = object.__new__(CheckpointManager)
    manager.checkpoint_dir = CanonicalWindowsPath(r"C:\Cache")
    monkeypatch.setattr(checkpoint, "Path", CanonicalWindowsPath)
    monkeypatch.setattr(checkpoint, "os", SimpleNamespace(path=ntpath))
    assert str(manager._resolve_checkpoint_path(value)) == expected
