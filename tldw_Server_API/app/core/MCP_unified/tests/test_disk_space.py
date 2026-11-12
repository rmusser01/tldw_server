from collections import namedtuple

import pytest

import tldw_Server_API.app.core.MCP_unified.modules.disk_space as disk_space_module
from tldw_Server_API.app.core.MCP_unified.modules.disk_space import get_free_disk_space_gb


def test_get_free_disk_space_gb_uses_statvfs(monkeypatch, tmp_path):
    statvfs_result = namedtuple("statvfs_result", ["f_bavail", "f_frsize"])
    monkeypatch.delattr(disk_space_module.os, "statvfs", raising=False)
    monkeypatch.setattr(
        disk_space_module.os,
        "statvfs",
        lambda _path: statvfs_result(1024, 4096),
        raising=False,
    )

    free_gb = get_free_disk_space_gb(tmp_path)

    assert free_gb == pytest.approx((1024 * 4096) / (1024 ** 3))


def test_get_free_disk_space_gb_falls_back_to_disk_usage(monkeypatch, tmp_path):
    usage_result = namedtuple("usage_result", ["total", "used", "free"])
    monkeypatch.delattr(disk_space_module.os, "statvfs", raising=False)

    def _raise_attr_error(_path):
        raise AttributeError("statvfs unavailable")

    monkeypatch.setattr(disk_space_module.os, "statvfs", _raise_attr_error, raising=False)
    monkeypatch.setattr(
        disk_space_module.shutil,
        "disk_usage",
        lambda _path: usage_result(100, 40, 60 * (1024 ** 3)),
    )

    free_gb = get_free_disk_space_gb(tmp_path)

    assert free_gb == pytest.approx(60.0)
