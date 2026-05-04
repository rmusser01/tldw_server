"""Bounded foreground indexing orchestration for native CodeGraph."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger

from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository

from .config import CodeGraphSettings
from .extractors.python_extractor import PythonAstExtractor
from .language_registry import CodeGraphLanguageRegistry
from .models import ExtractionResult


@dataclass(frozen=True)
class IndexingResult:
    """Result returned by foreground index and sync operations."""

    status: str
    counters: dict[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    """Discovered workspace-relative file with language and size metadata."""

    path: Path
    relative_path: str
    language_id: str
    stage: str
    size: int
    modified_at: float


@dataclass(frozen=True)
class _DiscoveryResult:
    """Candidate discovery output with an optional early terminal status."""

    candidates: list[_Candidate]
    status: str | None = None


class CodeGraphIndexer:
    """Bounded foreground file-inventory indexer."""

    def __init__(
        self,
        *,
        settings: CodeGraphSettings,
        registry: CodeGraphLanguageRegistry,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Create an indexer using settings, language metadata, and time source."""
        self._settings = settings
        self._registry = registry
        self._monotonic = monotonic or time.monotonic
        self._extractors = {"python": PythonAstExtractor()}

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
        """Run a bounded foreground full workspace index."""
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
        """Run a bounded foreground sync for the current workspace state."""
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
        """Execute shared index or sync flow with size and wall-clock safeguards."""
        del force  # Stage 1 records file inventory only; extraction cache comes later.
        repository.initialize()
        run_id = repository.record_index_run_start(workspace_key=workspace_key, mode=mode)
        counters = _empty_counters()
        errors: list[str] = []
        limit = max(1, int(max_files or self._settings.foreground_max_files))

        try:
            discovery = self._discover_candidates(
                workspace_root,
                languages=languages,
                counters=counters,
                max_files=limit,
            )
            if discovery.status is not None:
                repository.finish_index_run(
                    run_id,
                    status=discovery.status,
                    counters=counters,
                    error_summary=errors,
                )
                return IndexingResult(status=discovery.status, counters=counters, errors=tuple(errors))

            candidates = discovery.candidates
            foundation_candidates = [candidate for candidate in candidates if candidate.stage == "foundation"]
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
            discovered_paths = {candidate.relative_path for candidate in foundation_candidates}
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
                if self._is_binary_path(candidate.path):
                    counters["files_skipped"] += 1
                    continue

                if candidate.language_id in self._extractors:
                    source = candidate.path.read_bytes()
                    content_hash = self._hash_bytes(source)
                    extraction = self._extract(candidate, workspace_key, source)
                else:
                    content_hash = self._hash_file(candidate.path)
                    extraction = ExtractionResult()
                file_status = "indexed"
                if extraction.errors:
                    counters["errors"] += len(extraction.errors)
                    file_status = "extraction_failed"
                    errors.append(f"{candidate.relative_path}: {'; '.join(extraction.errors[:3])}")
                repository.upsert_file_and_replace_graph(
                    path=candidate.relative_path,
                    language=candidate.language_id,
                    size=candidate.size,
                    content_hash=content_hash,
                    modified_at=candidate.modified_at,
                    status=file_status,
                    errors=extraction.errors,
                    node_count=len(extraction.nodes),
                    nodes=extraction.nodes,
                    edges=extraction.edges,
                    unresolved_refs=extraction.unresolved_refs,
                )
                indexed_paths.add(candidate.relative_path)
                counters["files_indexed"] += 1

            if status == "complete" and not languages:
                repository.delete_missing_files(discovered_paths or indexed_paths)

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
        max_files: int,
    ) -> _DiscoveryResult:
        """Walk the workspace and collect supported-language candidate files."""
        root = workspace_root.expanduser().resolve(strict=False)
        selected_languages = set(languages or [])
        candidates: list[_Candidate] = []
        foundation_count = 0
        foundation_bytes = 0
        started_at = self._monotonic()

        for current_root, dir_names, file_names in os.walk(root):
            current_path = Path(current_root)
            dir_names[:] = [
                dirname
                for dirname in sorted(dir_names)
                if dirname not in self._settings.exclude_dirs
            ]

            for file_name in sorted(file_names):
                if self._monotonic() - started_at > self._settings.max_index_seconds:
                    return _DiscoveryResult(candidates=candidates, status="index_timed_out_for_foreground")

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
                    logger.debug(f"CodeGraph skipped unreadable path during discovery: {file_path} ({exc})")
                    counters["errors"] += 1
                    continue

                relative_path = resolved.relative_to(root).as_posix()
                counters["files_seen"] += 1
                if language.stage == "planned":
                    counters["planned_language_skipped"] += 1
                    counters["files_skipped"] += 1
                candidate = _Candidate(
                    path=resolved,
                    relative_path=relative_path,
                    language_id=language.language_id,
                    stage=language.stage,
                    size=int(stat_result.st_size),
                    modified_at=float(stat_result.st_mtime),
                )
                candidates.append(candidate)

                if candidate.stage == "foundation":
                    foundation_count += 1
                    foundation_bytes += candidate.size
                    if foundation_count > max_files or foundation_bytes > self._settings.foreground_max_bytes:
                        return _DiscoveryResult(candidates=candidates, status="index_too_large_for_foreground")
        return _DiscoveryResult(candidates=candidates)

    def _extract(self, candidate: _Candidate, workspace_key: str, source: bytes) -> ExtractionResult:
        """Run the language extractor for a candidate when one is implemented."""
        extractor = self._extractors.get(candidate.language_id)
        if extractor is None:
            return ExtractionResult()
        try:
            return extractor.extract(
                workspace_key=workspace_key,
                file_path=candidate.relative_path,
                source=source,
            )
        except ValueError as exc:
            logger.warning(f"CodeGraph extractor failed for {candidate.relative_path}: {exc}")
            return ExtractionResult(errors=(str(exc),))

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return a streaming SHA-256 content hash for a file path."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _hash_bytes(source: bytes) -> str:
        """Return a SHA-256 content hash for indexed file bytes."""
        return hashlib.sha256(source).hexdigest()

    @staticmethod
    def _is_binary_path(path: Path) -> bool:
        """Detect obvious binary files from a small leading byte probe."""
        with path.open("rb") as handle:
            return CodeGraphIndexer._is_binary(handle.read(4096))

    @staticmethod
    def _is_binary(source: bytes) -> bool:
        """Detect obvious binary files from the leading byte window."""
        return b"\x00" in source[:4096]


def _empty_counters() -> dict[str, int]:
    """Return a fresh counter map for a single index run."""
    return {
        "files_seen": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "files_too_large": 0,
        "planned_language_skipped": 0,
        "unsupported_language": 0,
        "errors": 0,
    }
