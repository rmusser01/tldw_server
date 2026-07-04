from __future__ import annotations

import os

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError, MacroValidationError
from tldw_Server_API.app.core.Chat_Macros import storage as storage_module
from tldw_Server_API.app.core.Chat_Macros.storage import ChatMacroStorage


def _macro_yaml(name: str = "daily_digest", command: str | None = None) -> str:
    return (
        "schema_version: 1\n"
        f"name: {name}\n"
        f"command: {command or name}\n"
        "steps:\n"
        "  - id: prompt\n"
        "    type: prompt\n"
        "    output: answer\n"
        "    prompt: Say hi.\n"
    )


def test_create_read_list_delete_macro_under_user_macros_directory(tmp_path):
    storage = ChatMacroStorage(tmp_path)

    created = storage.create("daily_digest", _macro_yaml(), {"notes.txt": "alpha"})

    assert (tmp_path / "macros" / "daily_digest" / "MACRO.yaml").is_file()
    assert created.definition.command == "daily_digest"
    assert created.supporting_files == {"notes.txt": "alpha"}
    assert [item.name for item in storage.list()] == ["daily_digest"]

    updated = storage.update("daily_digest", _macro_yaml(), {"notes.txt": "beta"})
    assert updated.digest != created.digest
    assert storage.read("daily_digest").supporting_files == {"notes.txt": "beta"}

    storage.delete("daily_digest")
    assert storage.list() == []


def test_storage_rejects_bad_names_and_supporting_file_traversal(tmp_path):
    storage = ChatMacroStorage(tmp_path)

    with pytest.raises(MacroValidationError, match="macro name"):
        storage.create("BadName", _macro_yaml())

    with pytest.raises(MacroValidationError, match="supporting file"):
        storage.create("daily_digest", _macro_yaml(), {"../escape.txt": "nope"})

    with pytest.raises(MacroValidationError, match="supporting file"):
        storage.create("daily_digest", _macro_yaml(), {"nested/file.txt": "nope"})


def test_storage_rejects_symlinked_macro_paths(tmp_path):
    storage = ChatMacroStorage(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "MACRO.yaml").write_text(_macro_yaml(), encoding="utf-8")

    macros_dir = tmp_path / "macros"
    os.symlink(outside, macros_dir / "daily_digest")

    with pytest.raises(MacroStorageError, match="symlink"):
        storage.read("daily_digest")

    (macros_dir / "daily_digest").unlink()
    macro_dir = macros_dir / "daily_digest"
    macro_dir.mkdir()
    os.symlink(outside / "MACRO.yaml", macro_dir / "MACRO.yaml")

    with pytest.raises(MacroStorageError, match="symlink"):
        storage.read("daily_digest")


def test_storage_rejects_symlinked_supporting_files(tmp_path):
    storage = ChatMacroStorage(tmp_path)
    storage.create("daily_digest", _macro_yaml(), {"notes.txt": "alpha"})
    notes = tmp_path / "macros" / "daily_digest" / "notes.txt"
    notes.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, notes)

    with pytest.raises(MacroStorageError, match="symlink"):
        storage.read("daily_digest")


def test_update_rejects_existing_symlinked_supporting_file(tmp_path):
    storage = ChatMacroStorage(tmp_path)
    storage.create("daily_digest", _macro_yaml(), {"notes.txt": "alpha"})
    notes = tmp_path / "macros" / "daily_digest" / "notes.txt"
    notes.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, notes)

    with pytest.raises(MacroStorageError, match="symlink"):
        storage.update("daily_digest", _macro_yaml("daily_digest", "team_digest"), {"notes.txt": "beta"})

    assert notes.is_symlink()
    assert (tmp_path / "macros" / "daily_digest" / "MACRO.yaml").read_text(encoding="utf-8") == _macro_yaml()


def test_yaml_only_update_rejects_existing_symlinked_supporting_file_before_write(tmp_path):
    storage = ChatMacroStorage(tmp_path)
    storage.create("daily_digest", _macro_yaml(), {"notes.txt": "alpha"})
    notes = tmp_path / "macros" / "daily_digest" / "notes.txt"
    notes.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, notes)

    with pytest.raises(MacroStorageError, match="symlink"):
        storage.update("daily_digest", _macro_yaml("daily_digest", "team_digest"))

    assert notes.is_symlink()
    assert (tmp_path / "macros" / "daily_digest" / "MACRO.yaml").read_text(encoding="utf-8") == _macro_yaml()


def test_failed_update_keeps_existing_macro_yaml(tmp_path, monkeypatch):
    storage = ChatMacroStorage(tmp_path)
    storage.create("daily_digest", _macro_yaml())

    def fail_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)

    with pytest.raises(MacroStorageError, match="failed to write"):
        storage.update("daily_digest", _macro_yaml("daily_digest", "team_digest"))

    assert storage.read("daily_digest").definition.command == "daily_digest"


def test_failed_supporting_file_update_keeps_existing_macro(tmp_path, monkeypatch):
    storage = ChatMacroStorage(tmp_path)
    storage.create("daily_digest", _macro_yaml(), {"notes.txt": "alpha"})
    original_replace = storage_module.os.replace

    def fail_supporting_file_replace(src, dst):
        if os.fspath(dst).endswith("notes.txt"):
            raise OSError("disk full")
        return original_replace(src, dst)

    monkeypatch.setattr(storage_module.os, "replace", fail_supporting_file_replace)

    with pytest.raises(MacroStorageError, match="failed to write"):
        storage.update("daily_digest", _macro_yaml("daily_digest", "team_digest"), {"notes.txt": "beta"})

    stored = storage.read("daily_digest")
    assert stored.definition.command == "daily_digest"
    assert stored.supporting_files == {"notes.txt": "alpha"}
