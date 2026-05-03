from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import CodeGraphSettings
from .language_registry import CodeGraphLanguageRegistry
from .repository import CodeGraphRepository


@dataclass(frozen=True)
class IndexingResult:
    status: str
    counters: dict[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    path: Path
    relative_path: str
    language_id: str
    stage: str
    size: int
    modified_at: float


class CodeGraphIndexer:
    """Bounded foreground file-inventory indexer."""

    def __init__(
        self,
        *,
        settings: CodeGraphSettings,
        registry: CodeGraphLanguageRegistry,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._monotonic = monotonic or time.monotonic

    def index_workspace(
        self,
        workspace_root: Path,
        workspace_key: str,
        repository: CodeGraphRepository,
        *,
        force: bool,
        languages: list[str] | tuple[str, ...] | None,
        max_files: int | None,
    ) -> IndexingResult:
        return self._run(
            workspace_root,
            workspace_key,
            repository,
            mode="foreground_index",
            force=force,
            languages=languages,
            max_files=max_files,
        )

    def sync_workspace(
        self,
        workspace_root: Path,
        workspace_key: str,
        repository: CodeGraphRepository,
        *,
        languages: list[str] | tuple[str, ...] | None,
        max_files: int | None,
    ) -> IndexingResult:
        return self._run(
            workspace_root,
            workspace_key,
            repository,
            mode="foreground_sync",
            force=False,
            languages=languages,
            max_files=max_files,
        )

    def _run(
        self,
        workspace_root: Path,
        workspace_key: str,
        repository: CodeGraphRepository,
        *,
        mode: str,
        force: bool,
        languages: list[str] | tuple[str, ...] | None,
        max_files: int | None,
    ) -> IndexingResult:
        del force  # Stage 1 records file inventory only; extraction cache comes later.
        repository.initialize()
        run_id = repository.record_index_run_start(workspace_key=workspace_key, mode=mode)
        counters = _empty_counters()
        errors: list[str] = []

        try:
            candidates = self._discover_candidates(workspace_root, languages=languages, counters=counters)
            foundation_candidates = [candidate for candidate in candidates if candidate.stage == "foundation"]
            limit = max(1, int(max_files or self._settings.foreground_max_files))
            total_bytes = sum(candidate.size for candidate in foundation_candidates)

            if len(foundation_candidates) > limit or total_bytes > self._settings.foreground_max_bytes:
                status = "index_too_large_for_foreground"
                repository.finish_index_run(
                    run_id,
                    status=status,
                    counters=counters,
                    error_summary=errors,
                )
                return IndexingResult(status=status, counters=counters, errors=tuple(errors))

            indexed_paths: set[str] = set()
            start = self._monotonic()
            status = "complete"

            for candidate in foundation_candidates:
                if self._monotonic() - start > self._settings.max_index_seconds:
                    status = "index_timed_out_for_foreground"
                    break
                if candidate.size > self._settings.max_file_size_bytes:
                    counters["files_too_large"] += 1
                    counters["files_skipped"] += 1
                    continue
                if self._is_binary(candidate.path):
                    counters["files_skipped"] += 1
                    continue

                content_hash = self._hash_file(candidate.path)
                repository.prepare_file_replacement(candidate.relative_path)
                repository.upsert_file(
                    path=candidate.relative_path,
                    language=candidate.language_id,
                    size=candidate.size,
                    content_hash=content_hash,
                    modified_at=candidate.modified_at,
                    status="indexed",
                    errors=[],
                )
                indexed_paths.add(candidate.relative_path)
                counters["files_indexed"] += 1

            if status == "complete":
                repository.delete_missing_files(indexed_paths)

            repository.finish_index_run(
                run_id,
                status=status,
                counters=counters,
                error_summary=errors,
            )
            return IndexingResult(status=status, counters=counters, errors=tuple(errors))
        except Exception as exc:
            errors.append(str(exc))
            counters["errors"] += 1
            repository.finish_index_run(
                run_id,
                status="error",
                counters=counters,
                error_summary=errors,
            )
            raise

    def _discover_candidates(
        self,
        workspace_root: Path,
        *,
        languages: list[str] | tuple[str, ...] | None,
        counters: dict[str, int],
    ) -> list[_Candidate]:
        root = workspace_root.expanduser().resolve(strict=False)
        selected_languages = set(languages or [])
        candidates: list[_Candidate] = []

        for current_root, dir_names, file_names in os.walk(root):
            current_path = Path(current_root)
            dir_names[:] = [
                dirname
                for dirname in sorted(dir_names)
                if dirname not in self._settings.exclude_dirs
            ]

            for file_name in sorted(file_names):
                file_path = current_path / file_name
                language = self._registry.language_for_path(file_path)
                if language is None:
                    counters["unsupported_language"] += 1
                    continue
                if selected_languages and language.language_id not in selected_languages:
                    counters["files_skipped"] += 1
                    continue

                try:
                    resolved = file_path.resolve(strict=False)
                    if resolved != root and root not in resolved.parents:
                        counters["files_skipped"] += 1
                        continue
                    stat_result = resolved.stat()
                except OSError as exc:
                    counters["errors"] += 1
                    continue

                relative_path = resolved.relative_to(root).as_posix()
                counters["files_seen"] += 1
                if language.stage == "planned":
                    counters["planned_language_skipped"] += 1
                    counters["files_skipped"] += 1
                candidates.append(
                    _Candidate(
                        path=resolved,
                        relative_path=relative_path,
                        language_id=language.language_id,
                        stage=language.stage,
                        size=int(stat_result.st_size),
                        modified_at=float(stat_result.st_mtime),
                    )
                )
        return candidates

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_binary(path: Path) -> bool:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(4096)


def _empty_counters() -> dict[str, int]:
    return {
        "files_seen": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "files_too_large": 0,
        "planned_language_skipped": 0,
        "unsupported_language": 0,
        "errors": 0,
    }
