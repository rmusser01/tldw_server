"""File-backed storage for user chat macro definitions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .exceptions import MacroNotFoundError, MacroStorageError, MacroValidationError
from .models import COMMAND_PATTERN, MacroDefinition
from .parser import load_macro_definition

MACRO_FILENAME = "MACRO.yaml"
MACRO_NAME_RE = re.compile(COMMAND_PATTERN)
SUPPORTING_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_MACRO_YAML_BYTES = 500_000
MAX_SUPPORTING_FILES_TOTAL_BYTES = 5 * 1024 * 1024


@dataclass(slots=True)
class StoredMacro:
    name: str
    definition: MacroDefinition
    raw: str
    digest: str
    supporting_files: dict[str, str]


class ChatMacroStorage:
    """CRUD for `macros/<name>/MACRO.yaml` under one user's base directory."""

    def __init__(self, user_base_path: Path | str) -> None:
        self.user_base_path = Path(user_base_path)
        self.macros_dir = self.user_base_path / "macros"
        self._ensure_macros_dir()

    def create(
        self,
        name: str,
        raw: str,
        supporting_files: dict[str, str | bytes] | None = None,
    ) -> StoredMacro:
        name = self._validate_macro_name(name)
        definition, raw_bytes, file_bytes = self._validate_payload(name, raw, supporting_files)
        macro_dir = self.macros_dir / name
        if macro_dir.exists() or macro_dir.is_symlink():
            raise MacroStorageError(f"macro already exists: {name}")
        macro_dir.mkdir(parents=False)
        self._replace_regular_file_no_follow(macro_dir / MACRO_FILENAME, raw_bytes)
        for filename, content in file_bytes.items():
            self._replace_regular_file_no_follow(macro_dir / filename, content)
        return self._stored(name, definition, raw, file_bytes)

    def update(
        self,
        name: str,
        raw: str,
        supporting_files: dict[str, str | bytes] | None = None,
    ) -> StoredMacro:
        name = self._validate_macro_name(name)
        macro_dir = self._existing_macro_dir(name)
        definition, raw_bytes, file_bytes = self._validate_payload(name, raw, supporting_files)
        existing_file_bytes = self._validate_existing_macro_files(macro_dir)

        if supporting_files is not None:
            self._replace_macro_directory(macro_dir, raw_bytes, file_bytes)
        else:
            self._replace_regular_file_no_follow(macro_dir / MACRO_FILENAME, raw_bytes)
            file_bytes = existing_file_bytes
        return self._stored(name, definition, raw, file_bytes)

    def read(self, name: str) -> StoredMacro:
        name = self._validate_macro_name(name)
        macro_dir = self._existing_macro_dir(name)
        raw_bytes = self._read_regular_file_bytes_no_follow(macro_dir / MACRO_FILENAME)
        if len(raw_bytes) > MAX_MACRO_YAML_BYTES:
            raise MacroStorageError("MACRO.yaml exceeds byte limit")
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MacroValidationError("MACRO.yaml must be UTF-8") from exc
        definition = load_macro_definition(raw)
        file_bytes = self._read_supporting_file_bytes(macro_dir)
        return self._stored(name, definition, raw, file_bytes)

    def delete(self, name: str) -> None:
        name = self._validate_macro_name(name)
        macro_dir = self._existing_macro_dir(name)
        shutil.rmtree(macro_dir)

    def list(self) -> list[StoredMacro]:
        if not self.macros_dir.exists():
            return []
        macros: list[StoredMacro] = []
        for path in sorted(self.macros_dir.iterdir(), key=lambda item: item.name):
            if path.name.startswith("."):
                continue
            macros.append(self.read(path.name))
        return macros

    def _validate_payload(
        self,
        name: str,
        raw: str,
        supporting_files: dict[str, str | bytes] | None,
    ) -> tuple[MacroDefinition, bytes, dict[str, bytes]]:
        raw_bytes = raw.encode("utf-8")
        if len(raw_bytes) > MAX_MACRO_YAML_BYTES:
            raise MacroValidationError("MACRO.yaml exceeds byte limit")
        definition = load_macro_definition(raw)
        if definition.name != name:
            raise MacroValidationError("macro name must match definition name")
        file_bytes = self._normalize_supporting_files(supporting_files or {})
        return definition, raw_bytes, file_bytes

    def _normalize_supporting_files(self, files: dict[str, str | bytes]) -> dict[str, bytes]:
        normalized: dict[str, bytes] = {}
        total = 0
        for filename, content in files.items():
            filename = self._validate_supporting_filename(filename)
            data = content if isinstance(content, bytes) else str(content).encode("utf-8")
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise MacroValidationError("supporting files must be UTF-8") from exc
            total += len(data)
            if total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                raise MacroValidationError("supporting files exceed total byte limit")
            normalized[filename] = data
        return normalized

    def _read_supporting_file_bytes(self, macro_dir: Path) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        total = 0
        for path in sorted(macro_dir.iterdir(), key=lambda item: item.name):
            if path.name == MACRO_FILENAME:
                continue
            self._validate_supporting_filename(path.name)
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                raise MacroStorageError(f"symlinked macro file is not allowed: {path}")
            if not stat.S_ISREG(entry.st_mode):
                raise MacroStorageError(f"supporting macro path is not a regular file: {path}")
            data = self._read_regular_file_bytes_no_follow(path)
            total += len(data)
            if total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                raise MacroStorageError("supporting files exceed total byte limit")
            files[path.name] = data
        return files

    def _stored(
        self,
        name: str,
        definition: MacroDefinition,
        raw: str,
        supporting_files: dict[str, bytes],
    ) -> StoredMacro:
        try:
            decoded = {filename: content.decode("utf-8") for filename, content in supporting_files.items()}
        except UnicodeDecodeError as exc:
            raise MacroValidationError("supporting files must be UTF-8") from exc
        return StoredMacro(
            name=name,
            definition=definition,
            raw=raw,
            digest=_digest(definition, supporting_files),
            supporting_files=decoded,
        )

    def _ensure_macros_dir(self) -> None:
        if self.macros_dir.exists() or self.macros_dir.is_symlink():
            entry = self.macros_dir.lstat()
            if stat.S_ISLNK(entry.st_mode):
                raise MacroStorageError(f"symlinked macros directory is not allowed: {self.macros_dir}")
            if not stat.S_ISDIR(entry.st_mode):
                raise MacroStorageError(f"macros path is not a directory: {self.macros_dir}")
            return
        self.macros_dir.mkdir(parents=True, exist_ok=True)

    def _existing_macro_dir(self, name: str) -> Path:
        macro_dir = self.macros_dir / name
        try:
            entry = macro_dir.lstat()
        except FileNotFoundError as exc:
            raise MacroNotFoundError(f"macro not found: {name}") from exc
        if stat.S_ISLNK(entry.st_mode):
            raise MacroStorageError(f"symlinked macro directory is not allowed: {macro_dir}")
        if not stat.S_ISDIR(entry.st_mode):
            raise MacroStorageError(f"macro path is not a directory: {macro_dir}")
        return macro_dir

    @staticmethod
    def _validate_macro_name(name: str) -> str:
        if not MACRO_NAME_RE.fullmatch(name or ""):
            raise MacroValidationError("invalid macro name")
        return name

    @staticmethod
    def _validate_supporting_filename(filename: str) -> str:
        if filename == MACRO_FILENAME or not SUPPORTING_FILE_RE.fullmatch(filename or ""):
            raise MacroValidationError("invalid supporting file name")
        if Path(filename).name != filename:
            raise MacroValidationError("invalid supporting file name")
        return filename

    @staticmethod
    def _read_regular_file_bytes_no_follow(path: Path) -> bytes:
        expected = path.lstat()
        if stat.S_ISLNK(expected.st_mode):
            raise MacroStorageError(f"symlinked macro file is not allowed: {path}")
        if not stat.S_ISREG(expected.st_mode):
            raise MacroStorageError(f"macro file is not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(expected, opened):
                raise MacroStorageError(f"macro file changed while being opened: {path}")
            with os.fdopen(fd, "rb", closefd=True) as file_obj:
                fd = -1
                return file_obj.read()
        finally:
            if fd >= 0:
                os.close(fd)

    def _validate_existing_macro_files(self, macro_dir: Path) -> dict[str, bytes]:
        self._read_regular_file_bytes_no_follow(macro_dir / MACRO_FILENAME)
        return self._read_supporting_file_bytes(macro_dir)

    def _replace_macro_directory(self, macro_dir: Path, raw_bytes: bytes, supporting_files: dict[str, bytes]) -> None:
        staging_path = Path(tempfile.mkdtemp(prefix=f".{macro_dir.name}.new.", dir=self.macros_dir))
        backup_path = Path(tempfile.mkdtemp(prefix=f".{macro_dir.name}.old.", dir=self.macros_dir))
        backup_path.rmdir()
        try:
            self._replace_regular_file_no_follow(staging_path / MACRO_FILENAME, raw_bytes)
            for filename, content in supporting_files.items():
                self._replace_regular_file_no_follow(staging_path / filename, content)

            self._existing_macro_dir(macro_dir.name)
            os.rename(macro_dir, backup_path)
            try:
                os.rename(staging_path, macro_dir)
            except OSError as exc:
                os.rename(backup_path, macro_dir)
                raise MacroStorageError(f"failed to publish macro directory: {macro_dir}") from exc
            shutil.rmtree(backup_path)
        except OSError as exc:
            raise MacroStorageError(f"failed to update macro directory: {macro_dir}") from exc
        finally:
            if staging_path.exists():
                shutil.rmtree(staging_path)

    @staticmethod
    def _replace_regular_file_no_follow(path: Path, data: bytes) -> None:
        if path.exists() or path.is_symlink():
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                raise MacroStorageError(f"symlinked macro file is not allowed: {path}")
            if not stat.S_ISREG(entry.st_mode):
                raise MacroStorageError(f"macro file is not a regular file: {path}")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb", closefd=True) as file_obj:
                fd = -1
                file_obj.write(data)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            raise MacroStorageError(f"failed to write macro file: {path}") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_path.exists():
                tmp_path.unlink()


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _digest(definition: MacroDefinition, supporting_files: dict[str, bytes]) -> str:
    payload = {
        "macro": definition.model_dump(mode="json"),
        "supporting_files": [
            {
                "name": filename,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for filename, content in sorted(supporting_files.items())
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
