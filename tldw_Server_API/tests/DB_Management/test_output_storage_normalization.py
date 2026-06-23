import os
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.services.outputs_service import normalize_output_storage_path


pytestmark = pytest.mark.unit


def _patch_outputs_dir(monkeypatch, base: Path) -> None:
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _uid: base)


def test_normalize_output_storage_path_accepts_safe_filename(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    assert normalize_output_storage_path(1, "report.md") == "report.md"


def test_normalize_output_storage_path_accepts_absolute_inside_base(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    storage_path = str(base / "report.md")
    assert normalize_output_storage_path(1, storage_path) == "report.md"


def test_normalize_output_storage_path_expands_user(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    storage_path = os.path.join("~", "outputs", "report.md")
    assert normalize_output_storage_path(1, storage_path) == "report.md"


@pytest.mark.parametrize(
    "storage_path",
    [
        str(Path("nested") / "report.md"),
        str(Path("..") / "report.md"),
        "report$.md",
    ],
)
def test_normalize_output_storage_path_rejects_invalid_relative(monkeypatch, tmp_path, storage_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    with pytest.raises(InvalidStoragePathError):
        normalize_output_storage_path(1, storage_path)


def test_normalize_output_storage_path_rejects_dotdot(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    with pytest.raises(InvalidStoragePathError):
        normalize_output_storage_path(1, "..")


def test_normalize_output_storage_path_rejects_absolute_outside_base(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    outside_path = str(tmp_path / "outside" / "report.md")
    with pytest.raises(InvalidStoragePathError):
        normalize_output_storage_path(1, outside_path)


def test_normalize_output_storage_path_rejects_absolute_nested_under_base(monkeypatch, tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    _patch_outputs_dir(monkeypatch, base)

    nested_path = str(base / "nested" / "report.md")
    with pytest.raises(InvalidStoragePathError):
        normalize_output_storage_path(1, nested_path)
