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


@pytest.mark.parametrize("leaf", [False, True, "root"])
def test_safe_join_rejects_links_even_when_destination_is_inside_root(tmp_path, leaf):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    base = link if leaf == "root" else tmp_path
    name = "." if leaf == "root" else ("link" if leaf else "link/file")
    assert path_utils.safe_join(str(base), name) is None


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


@pytest.mark.parametrize("name", [r"..\cache\Secret.txt", r"..\cache"])
def test_safe_join_rejects_distinct_canonical_windows_case_sibling(monkeypatch, name):
    windows_path = SimpleNamespace(**{
        key: getattr(ntpath, key)
        for key in ("isabs", "abspath", "join", "realpath", "relpath", "normcase")
    }, islink=lambda _: False)
    monkeypatch.setattr(path_utils, "os", SimpleNamespace(path=windows_path, sep="\\", pardir=".."))
    assert path_utils.safe_join(r"C:\Cache", name) is None


def test_safe_join_rejects_lexically_outside_windows_alias_before_realpath(monkeypatch):
    windows_path = SimpleNamespace(**{
        key: getattr(ntpath, key)
        for key in ("isabs", "abspath", "join", "relpath", "normcase")
    }, islink=lambda _: False)
    def unexpected_realpath(value):
        pytest.fail("lexically outside root alias must not be probed")

    windows_path.realpath = unexpected_realpath
    monkeypatch.setattr(path_utils, "os", SimpleNamespace(path=windows_path, sep="\\", pardir=".."))
    assert path_utils.safe_join(r"C:\Cache", r"..\CACHE\MixedCaseFile.txt") is None


def test_safe_join_rejects_lexical_escape_before_candidate_filesystem_probes(tmp_path, monkeypatch):
    base = tmp_path / "base"
    outside = tmp_path / "outside" / "secret"
    probes = []

    def record_probe(value):
        probes.append(os.fspath(value))
        return False

    paths = SimpleNamespace(**{
        key: getattr(os.path, key)
        for key in ("isabs", "abspath", "join", "relpath")
    }, islink=record_probe, realpath=lambda value: record_probe(value) or os.path.abspath(value))
    monkeypatch.setattr(path_utils, "os", SimpleNamespace(path=paths, sep=os.sep, pardir=os.pardir))
    assert path_utils.safe_join(str(base), "../outside/secret") is None
    assert str(outside) not in probes


def test_safe_join_does_not_probe_children_beneath_directory_link(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    parent_link = base / "linked"
    parent_link.symlink_to(outside, target_is_directory=True)
    probes = []

    def record_probe(function, value):
        probes.append(os.fspath(value))
        return function(value)

    paths = SimpleNamespace(**{
        key: getattr(os.path, key)
        for key in ("isabs", "abspath", "join", "relpath")
    }, islink=lambda value: record_probe(os.path.islink, value),
        realpath=lambda value: record_probe(os.path.realpath, value))
    monkeypatch.setattr(path_utils, "os", SimpleNamespace(path=paths, sep=os.sep, pardir=os.pardir))
    assert path_utils.safe_join(str(base), "linked/secret") is None
    assert str(parent_link / "secret") not in probes
