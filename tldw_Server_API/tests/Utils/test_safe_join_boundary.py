"""Containment contracts shared by storage roots, directories and file leaves."""
import ntpath
import os
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Utils import path_utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", [".", "nested/..", "nested/file", "file"])
def test_safe_join_preserves_confined_paths(tmp_path, name):
    assert path_utils.safe_join(str(tmp_path), name) == os.path.realpath(tmp_path / name)


@pytest.mark.parametrize("name", ["../base-sibling/file", "../outside", "/absolute"])
def test_safe_join_rejects_lexical_escapes(tmp_path, name):
    assert path_utils.safe_join(str(tmp_path / "base"), name) is None


def test_safe_join_accepts_filesystem_root_and_equal_root():
    assert path_utils.safe_join(os.path.abspath(os.sep), ".") == os.path.abspath(os.sep)
    assert path_utils.safe_join(os.path.abspath(os.sep), "child") == os.path.join(os.path.abspath(os.sep), "child")


@pytest.mark.parametrize("leaf", [False, True])
def test_safe_join_rejects_links_even_when_destination_is_inside_root(tmp_path, leaf):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    assert path_utils.safe_join(str(tmp_path), "link" if leaf else "link/file") is None


@pytest.mark.parametrize("base,name,expected", [
    (r"C:\Cache", ".", r"c:\cache"),
    (r"C:\Cache", r"Nested\Model", r"c:\cache\nested\model"),
    ("C:" + ntpath.sep, "child", r"c:\child"),
    (r"C:\Cache", r"..\CacheSibling\file", None),
    (r"C:\Cache", r"D:\Elsewhere\file", None),
    (r"\\Server\Share\Cache", r"Nested\Model", r"\\server\share\cache\nested\model"),
    (r"\\Server\Share", ".", r"\\server\share" + ntpath.sep),
    (r"\\Server\Share", r"\\Other\Share\file", None),
])
def test_safe_join_windows_drive_case_and_unc_contract(monkeypatch, base, name, expected):
    # Exercise Windows path semantics without requiring a Windows filesystem.
    windows_path = SimpleNamespace(**{
        key: getattr(ntpath, key)
        for key in ("isabs", "abspath", "join", "realpath", "commonpath", "relpath", "normcase")
    }, islink=lambda _: False)
    monkeypatch.setattr(path_utils, "os", SimpleNamespace(path=windows_path, sep="\\", pardir=".."))
    result = path_utils.safe_join(base, name)
    assert (ntpath.normcase(result) if result is not None else None) == expected
    if result is not None:
        assert result == ntpath.realpath(ntpath.abspath(ntpath.join(base, name)))
