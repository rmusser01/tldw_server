from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable in this environment: {exc}")


def _mkfifo_or_skip(path: Path) -> None:
    mkfifo = getattr(os, "mkfifo", None)
    if mkfifo is None:
        pytest.skip("os.mkfifo is unavailable in this environment")
    try:
        mkfifo(path)
    except OSError as exc:
        pytest.skip(f"FIFOs are unavailable in this environment: {exc}")


def _fail_if_fifo_opened(monkeypatch: pytest.MonkeyPatch, inventory_module: object, fifo: Path) -> None:
    original_open = os.open

    def guarded_open(path: object, flags: int, mode: int = 0o777, *args: object, **kwargs: object) -> int:
        if Path(path).name == fifo.name:
            pytest.fail(f"FIFO was opened during inventory: {path}")
        return original_open(path, flags, mode, *args, **kwargs)

    supports_dir_fd = set(inventory_module.os.supports_dir_fd)
    supports_dir_fd.add(guarded_open)
    monkeypatch.setattr(inventory_module.os, "supports_dir_fd", supports_dir_fd)
    monkeypatch.setattr(inventory_module.os, "open", guarded_open)


def _raise_child_stat_for_name(
    monkeypatch: pytest.MonkeyPatch,
    inventory_module: object,
    protected_name: str,
) -> None:
    original_child_stat = getattr(inventory_module, "_stat_child_no_follow", None)

    def guarded_child_stat(*args: object, **kwargs: object) -> os.stat_result:
        name = kwargs.get("name")
        if name is None and len(args) >= 2:
            name = args[1]
        if name == protected_name:
            raise PermissionError("simulated permission denied")
        if original_child_stat is not None:
            return original_child_stat(*args, **kwargs)
        return os.stat_result((0,) * 10)

    monkeypatch.setattr(inventory_module, "_stat_child_no_follow", guarded_child_stat, raising=False)


def _raise_fd_listdir_type_error(monkeypatch: pytest.MonkeyPatch, inventory_module: object) -> None:
    original_listdir = inventory_module.os.listdir

    def guarded_listdir(path: object) -> list[str]:
        if isinstance(path, int):
            raise TypeError("simulated fd listdir failure")
        return original_listdir(path)

    supports_fd = set(inventory_module.os.supports_fd)
    supports_fd.add(guarded_listdir)
    monkeypatch.setattr(inventory_module.os, "supports_fd", supports_fd)
    monkeypatch.setattr(inventory_module.os, "listdir", guarded_listdir)


def _fail_resolve_for_path(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    original_resolve = Path.resolve

    def guarded_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        try:
            self.relative_to(root)
        except ValueError:
            return original_resolve(self, *args, **kwargs)
        pytest.fail(f"Path.resolve was called during symlink reporting: {self}")

    monkeypatch.setattr(Path, "resolve", guarded_resolve)


def _fail_parent_traversal_lstat(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    original_lstat = Path.lstat

    def guarded_lstat(self: Path) -> os.stat_result:
        try:
            self.relative_to(root)
        except ValueError:
            return original_lstat(self)
        if ".." in self.parts:
            pytest.fail(f"Path.lstat was called before rejecting parent traversal: {self}")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", guarded_lstat)


def _skip_if_fd_traversal_unavailable() -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    try:
        inventory._require_fd_traversal_support()
    except OSError as exc:
        pytest.skip(f"fd-relative traversal is unavailable in this environment: {exc}")


def test_inventory_user_skill_directory_includes_supporting_files(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill_dir / "ref.md").write_text("reference", encoding="utf-8")

    assets = inventory_user_skills(user_id=1, skills_root=tmp_path / "skills")

    assert len(assets) == 1
    assert assets[0].asset_id == "skill:user:1/demo"
    assert assets[0].executable is True
    assert assets[0].source_type == "skill_file"


def test_inventory_user_skills_accepts_positional_public_arguments(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")

    assets = inventory_user_skills(1, tmp_path / "skills")

    assert [asset.asset_id for asset in assets] == ["skill:user:1/demo"]


def test_inventory_user_skill_digest_changes_when_supporting_file_changes(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_user_skills

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    reference = skill_dir / "ref.md"
    reference.write_text("reference", encoding="utf-8")

    original = inventory_user_skills(user_id=1, skills_root=tmp_path / "skills")
    reference.write_text("changed reference", encoding="utf-8")
    changed = inventory_user_skills(user_id=1, skills_root=tmp_path / "skills")

    assert original[0].digest != changed[0].digest


def test_inventory_user_skill_reports_symlink_escape_without_hashing_target(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside secret", encoding="utf-8")
    _symlink_or_skip(skill_dir / "escape.md", outside)

    first = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)
    outside.write_text("changed outside secret", encoding="utf-8")
    second = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert len(first.assets) == 1
    assert first.assets[0].digest == second.assets[0].digest
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_user_skill_broken_symlink_child_does_not_resolve_target(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    _symlink_or_skip(skill_dir / "missing.md", tmp_path / "missing.md")
    _fail_resolve_for_path(monkeypatch, tmp_path)

    result = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert [asset.asset_id for asset in result.assets] == ["skill:user:1/demo"]
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_user_skill_reports_symlinked_skill_file_without_hashing_target(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo"
    skill_dir.mkdir(parents=True)
    outside = tmp_path / "outside_skill.md"
    outside.write_text("---\nname: outside\n---\nOutside", encoding="utf-8")
    _symlink_or_skip(skill_dir / "SKILL.md", outside)

    first = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)
    outside.write_text("changed outside skill", encoding="utf-8")
    second = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_user_skill_symlinked_root_reports_error_without_hashing_target(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    outside_root = tmp_path / "outside_skills"
    skill_dir = outside_root / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    skills_root = tmp_path / "skills"
    _symlink_or_skip(skills_root, outside_root, target_is_directory=True)

    first = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)
    (skill_dir / "SKILL.md").write_text("changed outside skill", encoding="utf-8")
    second = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_symlinked_root_parent_reports_error_without_hashing_target(
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    outside_parent = tmp_path / "outside_parent"
    skill_dir = outside_parent / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    linked_parent = tmp_path / "linked_parent"
    _symlink_or_skip(linked_parent, outside_parent, target_is_directory=True)
    skills_root = linked_parent / "skills"

    first = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)
    (skill_dir / "SKILL.md").write_text("changed outside skill", encoding="utf-8")
    second = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_broken_symlinked_root_parent_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    linked_parent = tmp_path / "linked_parent"
    _symlink_or_skip(linked_parent, tmp_path / "missing_parent", target_is_directory=True)
    skills_root = linked_parent / "skills"

    result = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_parent_traversal_root_reports_error_before_lstat(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    root = tmp_path / "root"
    skill_dir = root / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (root / "child").mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    _fail_parent_traversal_lstat(monkeypatch, tmp_path)

    result = inventory_user_skills_with_findings(
        user_id=1,
        skills_root=root / "child" / ".." / "skills",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_broken_symlink_root_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    skills_root = tmp_path / "skills"
    _symlink_or_skip(skills_root, tmp_path / "missing_skills", target_is_directory=True)

    result = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_file_root_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_user_skills_with_findings,
    )

    skills_root = tmp_path / "skills"
    skills_root.write_text("not a directory", encoding="utf-8")

    result = inventory_user_skills_with_findings(user_id=1, skills_root=skills_root)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_reports_no_follow_open_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    original_open_child_dir = inventory._open_child_dir_no_follow_fd

    def raise_open_error(dir_fd: int, name: str, path: Path) -> int:
        if path == skill_dir:
            raise OSError("simulated no-follow failure")
        return original_open_child_dir(dir_fd, name, path)

    monkeypatch.setattr(inventory, "_open_child_dir_no_follow_fd", raise_open_error, raising=False)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_user_skill_root_fd_open_failure_reports_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")

    def raise_open_error(*args: object, **kwargs: object) -> int:
        raise OSError("simulated unsupported fd support")

    monkeypatch.setattr(inventory, "_open_dir_no_follow_fd", raise_open_error, raising=False)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_root_fd_listdir_type_error_reports_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    _raise_fd_listdir_type_error(monkeypatch, inventory)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1"


def test_inventory_user_skill_symlink_preflight_permission_error_reports_finding(
    monkeypatch,
    tmp_path,
) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    _raise_child_stat_for_name(monkeypatch, inventory, skill_file.name)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_user_skill_reads_files_without_path_reader(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill_dir / "ref.md").write_text("reference", encoding="utf-8")
    fd_reader_paths: list[Path] = []
    original_fd_reader = inventory._read_regular_file_from_dir_fd

    def is_inventory_path(path: Path) -> bool:
        try:
            path.relative_to(skill_dir)
        except ValueError:
            return False
        return True

    def fail_path_read_bytes(self: Path) -> bytes:
        if is_inventory_path(self):
            pytest.fail(f"path-based read_bytes was called for {self}")
        return original_path_read_bytes(self)

    def fail_path_open(self: Path, *args: object, **kwargs: object) -> object:
        if is_inventory_path(self):
            pytest.fail(f"path-based open was called for {self}")
        return original_path_open(self, *args, **kwargs)

    def counting_fd_reader(*args: object, **kwargs: object) -> bytes | None:
        fd_reader_paths.append(kwargs["path"])
        return original_fd_reader(*args, **kwargs)

    original_path_read_bytes = Path.read_bytes
    original_path_open = Path.open
    monkeypatch.setattr(Path, "read_bytes", fail_path_read_bytes)
    monkeypatch.setattr(Path, "open", fail_path_open)
    monkeypatch.setattr(inventory, "_read_regular_file_from_dir_fd", counting_fd_reader)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert [asset.asset_id for asset in result.assets] == ["skill:user:1/demo"]
    assert result.findings == ()
    assert {path.name for path in fd_reader_paths} == {"SKILL.md", "ref.md"}


def test_inventory_user_skills_falls_back_when_fd_traversal_is_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    monkeypatch.setattr(inventory.os, "supports_fd", set())

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert [asset.asset_id for asset in result.assets] == ["skill:user:1/demo"]
    assert result.findings == ()


def test_inventory_user_skill_fifo_supporting_file_reports_error_without_opening(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    fifo = skill_dir / "pipe.md"
    _mkfifo_or_skip(fifo)
    _fail_if_fifo_opened(monkeypatch, inventory, fifo)

    result = inventory.inventory_user_skills_with_findings(
        user_id=1,
        skills_root=tmp_path / "skills",
    )

    assert [asset.asset_id for asset in result.assets] == ["skill:user:1/demo"]
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "skill:user:1/demo"


def test_inventory_prompt_files_finds_supported_extensions(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_prompt_files

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: prompt", encoding="utf-8")
    (prompts / "ignore.bin").write_bytes(b"no")

    assets = inventory_prompt_files(prompts_dir=prompts)

    assert [asset.asset_id for asset in assets] == ["prompt_file:rag.prompts.yaml"]
    assert assets[0].executable is False


def test_inventory_prompt_files_accepts_positional_public_argument(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import inventory_prompt_files

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: prompt", encoding="utf-8")

    assets = inventory_prompt_files(prompts)

    assert [asset.asset_id for asset in assets] == ["prompt_file:rag.prompts.yaml"]


def test_inventory_prompt_files_does_not_recurse_or_follow_symlink_escape(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")
    nested = prompts / "nested"
    nested.mkdir()
    (nested / "nested.md").write_text("nested prompt", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside prompt", encoding="utf-8")
    _symlink_or_skip(prompts / "escape.md", outside)

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert [asset.asset_id for asset in result.assets] == ["prompt_file:root.md"]
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:escape.md"


def test_inventory_prompt_files_broken_symlink_child_does_not_resolve_target(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    _symlink_or_skip(prompts / "missing.md", tmp_path / "missing.md")
    _fail_resolve_for_path(monkeypatch, tmp_path)

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:missing.md"


def test_inventory_prompt_files_reports_symlinked_directory_escape(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    outside = tmp_path / "outside_prompts"
    outside.mkdir()
    (outside / "outside.md").write_text("outside prompt", encoding="utf-8")
    _symlink_or_skip(prompts / "linked_prompts", outside)

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:linked_prompts"


def test_inventory_prompt_files_symlinked_root_reports_error_without_hashing_target(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    outside_prompts = tmp_path / "outside_prompts"
    outside_prompts.mkdir()
    (outside_prompts / "root.md").write_text("outside prompt", encoding="utf-8")
    prompts = tmp_path / "Prompts"
    _symlink_or_skip(prompts, outside_prompts, target_is_directory=True)

    first = inventory_prompt_files_with_findings(prompts_dir=prompts)
    (outside_prompts / "root.md").write_text("changed outside prompt", encoding="utf-8")
    second = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_symlinked_root_parent_reports_error_without_hashing_target(
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    outside_parent = tmp_path / "outside_parent"
    outside_prompts = outside_parent / "Prompts"
    outside_prompts.mkdir(parents=True)
    (outside_prompts / "root.md").write_text("outside prompt", encoding="utf-8")
    linked_parent = tmp_path / "linked_parent"
    _symlink_or_skip(linked_parent, outside_parent, target_is_directory=True)
    prompts = linked_parent / "Prompts"

    first = inventory_prompt_files_with_findings(prompts_dir=prompts)
    (outside_prompts / "root.md").write_text("changed outside prompt", encoding="utf-8")
    second = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert first.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_broken_symlinked_root_parent_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    linked_parent = tmp_path / "linked_parent"
    _symlink_or_skip(linked_parent, tmp_path / "missing_parent", target_is_directory=True)
    prompts = linked_parent / "Prompts"

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_parent_traversal_root_reports_error_before_lstat(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    root = tmp_path / "root"
    prompts = root / "Prompts"
    prompts.mkdir(parents=True)
    (root / "child").mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")
    _fail_parent_traversal_lstat(monkeypatch, tmp_path)

    result = inventory_prompt_files_with_findings(
        prompts_dir=root / "child" / ".." / "Prompts",
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_broken_symlink_root_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    prompts = tmp_path / "Prompts"
    _symlink_or_skip(prompts, tmp_path / "missing_prompts", target_is_directory=True)

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_file_root_reports_error(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )

    prompts = tmp_path / "Prompts"
    prompts.write_text("not a directory", encoding="utf-8")

    result = inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_reports_no_follow_open_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")

    def raise_open_error(*args: object, **kwargs: object) -> int:
        raise OSError("simulated no-follow failure")

    monkeypatch.setattr(inventory, "_open_file_no_follow_fd", raise_open_error, raising=False)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:root.md"


def test_inventory_prompt_files_root_fd_open_failure_reports_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")

    def raise_open_error(*args: object, **kwargs: object) -> int:
        raise OSError("simulated unsupported fd support")

    monkeypatch.setattr(inventory, "_open_dir_no_follow_fd", raise_open_error, raising=False)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_root_fd_listdir_type_error_reports_error(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")
    _raise_fd_listdir_type_error(monkeypatch, inventory)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:Prompts"


def test_inventory_prompt_files_symlink_preflight_permission_error_reports_finding(
    monkeypatch,
    tmp_path,
) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    prompt_file = prompts / "root.md"
    prompt_file.write_text("root prompt", encoding="utf-8")
    _raise_child_stat_for_name(monkeypatch, inventory, prompt_file.name)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:root.md"


def test_inventory_prompt_files_reads_files_without_path_reader(monkeypatch, tmp_path) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")
    fd_reader_paths: list[Path] = []
    original_fd_reader = inventory._read_regular_file_from_dir_fd

    def is_inventory_path(path: Path) -> bool:
        try:
            path.relative_to(prompts)
        except ValueError:
            return False
        return True

    def fail_path_read_bytes(self: Path) -> bytes:
        if is_inventory_path(self):
            pytest.fail(f"path-based read_bytes was called for {self}")
        return original_path_read_bytes(self)

    def fail_path_open(self: Path, *args: object, **kwargs: object) -> object:
        if is_inventory_path(self):
            pytest.fail(f"path-based open was called for {self}")
        return original_path_open(self, *args, **kwargs)

    def counting_fd_reader(*args: object, **kwargs: object) -> bytes | None:
        fd_reader_paths.append(kwargs["path"])
        return original_fd_reader(*args, **kwargs)

    original_path_read_bytes = Path.read_bytes
    original_path_open = Path.open
    monkeypatch.setattr(Path, "read_bytes", fail_path_read_bytes)
    monkeypatch.setattr(Path, "open", fail_path_open)
    monkeypatch.setattr(inventory, "_read_regular_file_from_dir_fd", counting_fd_reader)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert [asset.asset_id for asset in result.assets] == ["prompt_file:root.md"]
    assert result.findings == ()
    assert [path.name for path in fd_reader_paths] == ["root.md"]


def test_inventory_prompt_files_falls_back_when_fd_traversal_is_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "root.md").write_text("root prompt", encoding="utf-8")
    (prompts / "ignore.bin").write_bytes(b"no")
    monkeypatch.setattr(inventory.os, "supports_fd", set())

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert [asset.asset_id for asset in result.assets] == ["prompt_file:root.md"]
    assert result.findings == ()


def test_inventory_prompt_files_fifo_reports_error_without_opening(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    fifo = prompts / "pipe.md"
    _mkfifo_or_skip(fifo)
    _fail_if_fifo_opened(monkeypatch, inventory, fifo)

    result = inventory.inventory_prompt_files_with_findings(prompts_dir=prompts)

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:pipe.md"


def test_inventory_env_prompt_overrides_finds_configured_files(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides,
    )

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")

    assets = inventory_env_prompt_overrides(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert [asset.asset_id for asset in assets] == [
        "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    ]
    assert assets[0].metadata["source_label"] == "env:TLDW_PROMPT_FILE_CHAT__SYSTEM"


def test_inventory_env_prompt_overrides_falls_back_when_fd_traversal_is_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")
    monkeypatch.setattr(inventory.os, "supports_fd", set())

    result = inventory.inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert [asset.asset_id for asset in result.assets] == [
        "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    ]
    assert result.findings == ()


def test_inventory_env_prompt_overrides_accepts_positional_public_argument(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides,
    )

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")

    assets = inventory_env_prompt_overrides(
        {"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert [asset.asset_id for asset in assets] == [
        "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    ]


def test_inventory_env_prompt_overrides_reports_symlink_without_hashing_target(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides_with_findings,
    )

    outside = tmp_path / "outside_override.md"
    outside.write_text("outside prompt", encoding="utf-8")
    override = tmp_path / "override.md"
    _symlink_or_skip(override, outside)

    first = inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )
    outside.write_text("changed outside prompt", encoding="utf-8")
    second = inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert (
        first.findings[0].asset_id
        == "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    )


def test_inventory_env_prompt_overrides_reports_symlinked_parent_without_hashing_target(
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides_with_findings,
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    override = outside / "override.md"
    override.write_text("outside prompt", encoding="utf-8")
    link = tmp_path / "linked_parent"
    _symlink_or_skip(link, outside, target_is_directory=True)

    first = inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(link / "override.md")}
    )
    override.write_text("changed outside prompt", encoding="utf-8")
    second = inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(link / "override.md")}
    )

    assert first.assets == second.assets == ()
    assert [finding.state for finding in first.findings] == ["verification_error"]
    assert (
        first.findings[0].asset_id
        == "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    )


def test_inventory_env_prompt_overrides_fifo_reports_error_without_opening(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity import inventory

    fifo = tmp_path / "pipe.md"
    _mkfifo_or_skip(fifo)
    _fail_if_fifo_opened(monkeypatch, inventory, fifo)

    result = inventory.inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(fifo)}
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert (
        result.findings[0].asset_id
        == "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:pipe.md"
    )


def test_inventory_env_prompt_overrides_symlink_preflight_permission_error_reports_finding(
    monkeypatch,
    tmp_path,
) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")
    _raise_child_stat_for_name(monkeypatch, inventory, override.name)

    result = inventory.inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert (
        result.findings[0].asset_id
        == "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    )


def test_inventory_env_prompt_overrides_parent_fd_open_failure_reports_error(
    monkeypatch,
    tmp_path,
) -> None:
    _skip_if_fd_traversal_unavailable()
    from tldw_Server_API.app.core.Context_Integrity import inventory

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")

    def raise_open_error(*args: object, **kwargs: object) -> int:
        raise OSError("simulated unsupported fd support")

    monkeypatch.setattr(inventory, "_open_dir_no_follow_fd", raise_open_error, raising=False)

    result = inventory.inventory_env_prompt_overrides_with_findings(
        environ={"TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override)}
    )

    assert result.assets == ()
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert (
        result.findings[0].asset_id
        == "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    )


def test_inventory_env_prompt_overrides_reports_missing_and_ignores_other_vars(tmp_path) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides_with_findings,
    )

    override = tmp_path / "override.md"
    override.write_text("override prompt", encoding="utf-8")
    missing = tmp_path / "missing.md"

    result = inventory_env_prompt_overrides_with_findings(
        environ={
            "OTHER_PROMPT_FILE": str(missing),
            "TLDW_PROMPT_FILE_EMPTY": "  ",
            "TLDW_PROMPT_FILE_CHAT__SYSTEM": str(override),
            "TLDW_PROMPT_FILE_MISSING": str(missing),
        }
    )

    assert [asset.asset_id for asset in result.assets] == [
        "prompt_file:env:TLDW_PROMPT_FILE_CHAT__SYSTEM:override.md"
    ]
    assert [finding.state for finding in result.findings] == ["verification_error"]
    assert result.findings[0].asset_id == "prompt_file:env:TLDW_PROMPT_FILE_MISSING:missing.md"
