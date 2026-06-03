from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy
from tldw_Server_API.app.core.Workspaces.file_inventory_scanner import (
    InventoryScanBounds,
    scan_workspace_file_inventory,
)


def test_scanner_records_relative_file_and_directory_metadata(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    items = {item["relative_path"]: item for item in result.items}
    assert sorted(items) == ["README.md", "src", "src/app.py"]
    assert items["src"]["entry_kind"] == "directory"
    assert items["src/app.py"]["entry_kind"] == "file"
    assert items["src/app.py"]["extension"] == ".py"
    assert items["src/app.py"]["size_bytes"] == len("print('hello')\n")
    assert all(not item["relative_path"].startswith("/") for item in result.items)
    assert result.counts["files"] == 2
    assert result.counts["directories"] == 1
    assert result.counts["total_entries"] == 3
    assert result.coverage_complete is True


def test_scanner_does_not_open_ordinary_file_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notes.txt").write_text("do not read me", encoding="utf-8")

    def fail_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("scanner opened an ordinary file")

    monkeypatch.setattr(builtins, "open", fail_open)

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    assert [item["relative_path"] for item in result.items] == ["notes.txt"]
    assert result.coverage_complete is True


def test_scanner_records_symlink_entries_without_following_directory_targets(tmp_path: Path) -> None:
    target = tmp_path / "real-dir"
    target.mkdir()
    (target / "nested.txt").write_text("hidden through symlink", encoding="utf-8")
    link = tmp_path / "linked-dir"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    items = {item["relative_path"]: item for item in result.items}
    assert items["linked-dir"]["entry_kind"] == "symlink"
    assert "linked-dir/nested.txt" not in items
    assert "real-dir/nested.txt" in items
    assert result.counts["symlinks"] == 1


def test_scanner_rejects_symlink_root_without_following_target(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    (real_root / "secret.txt").write_text("do not traverse", encoding="utf-8")
    link_root = tmp_path / "link-root"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    result = scan_workspace_file_inventory(
        link_root,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    assert result.items == ()
    assert result.coverage_complete is False
    assert result.diagnostics[0]["code"] == "root_symlink_not_supported"
    assert "secret.txt" not in str(result)


def test_scanner_honors_builtin_ignore_policy_without_emitting_ignored_items(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = {}", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    relative_paths = {item["relative_path"] for item in result.items}
    assert "node_modules" not in relative_paths
    assert "node_modules/pkg/index.js" not in relative_paths
    assert ".env.local" not in relative_paths
    assert "src/app.py" in relative_paths
    assert result.counts["ignored"] == 2


def test_scanner_records_partial_diagnostics_for_directory_listing_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Workspaces.file_inventory_scanner as scanner

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "secret.txt").write_text("hidden", encoding="utf-8")
    real_scandir = scanner.os.scandir

    def fake_scandir(path: str | bytes | Path) -> object:
        if Path(path).name == "blocked":
            raise PermissionError("permission denied")
        return real_scandir(path)

    monkeypatch.setattr(scanner.os, "scandir", fake_scandir)

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    assert result.coverage_complete is False
    assert result.diagnostics[0]["code"] == "permission_denied"
    assert result.diagnostics[0]["path_hint"] == "blocked"
    assert "/" not in result.diagnostics[0]["message"]
    assert not str(tmp_path) in str(result.diagnostics)


def test_scanner_stops_at_file_limit_and_reports_bounded_partial_result(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"file-{index}.txt").write_text(str(index), encoding="utf-8")

    result = scan_workspace_file_inventory(
        tmp_path,
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(max_files=2, max_diagnostics=1),
    )

    assert result.counts["files"] == 2
    assert result.counts["total_entries"] == 2
    assert result.coverage_complete is False
    assert [diagnostic["code"] for diagnostic in result.diagnostics] == ["scan_limit_reached"]
    assert all(not item["relative_path"].startswith("/") for item in result.items)


def test_scanner_redacts_absolute_paths_from_diagnostics(tmp_path: Path) -> None:
    result = scan_workspace_file_inventory(
        tmp_path / "missing",
        policy=build_inventory_ignore_policy(),
        bounds=InventoryScanBounds(),
    )

    assert result.coverage_complete is False
    assert result.diagnostics[0]["code"] == "root_not_directory"
    assert str(tmp_path) not in str(result.diagnostics)
