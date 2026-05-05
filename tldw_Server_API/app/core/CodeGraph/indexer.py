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
from .extractors.c_family_extractor import CppTreeSitterExtractor, CTreeSitterExtractor
from .extractors.csharp_extractor import CSharpTreeSitterExtractor
from .extractors.java_extractor import JavaTreeSitterExtractor
from .extractors.javascript_extractor import JavaScriptTreeSitterExtractor
from .extractors.kotlin_extractor import KotlinTreeSitterExtractor
from .extractors.python_extractor import PythonAstExtractor
from .extractors.tree_sitter_loader import load_parser
from .extractors.typescript_extractor import TypeScriptTreeSitterExtractor
from .language_registry import CodeGraphLanguageRegistry
from .models import ExtractionResult
from .resolver import CodeGraphReferenceResolver


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
    symbol_extraction: bool
    size: int
    modified_at: float


@dataclass(frozen=True)
class _DiscoveryResult:
    """Candidate discovery output with an optional early terminal status."""

    candidates: list[_Candidate]
    status: str | None = None


@dataclass(frozen=True)
class _CandidateContent:
    """Opened file content or hash derived from a single file stream."""

    content_hash: str
    source: bytes | None = None
    is_binary: bool = False


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
        if load_parser("javascript").available:
            self._extractors["javascript"] = JavaScriptTreeSitterExtractor()
        if load_parser("typescript").available:
            self._extractors["typescript"] = TypeScriptTreeSitterExtractor()
        if load_parser("java").available:
            self._extractors["java"] = JavaTreeSitterExtractor()
        if load_parser("kotlin").available:
            self._extractors["kotlin"] = KotlinTreeSitterExtractor()
        if load_parser("csharp").available:
            self._extractors["csharp"] = CSharpTreeSitterExtractor()
        if load_parser("c").available:
            self._extractors["c"] = CTreeSitterExtractor()
        if load_parser("cpp").available:
            self._extractors["cpp"] = CppTreeSitterExtractor()

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
            foundation_candidates = [
                candidate
                for candidate in candidates
                if candidate.stage == "foundation" and candidate.symbol_extraction
            ]
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

                has_extractor = candidate.language_id in self._extractors
                try:
                    content = self._read_candidate_content(candidate, needs_source=has_extractor)
                except OSError as exc:
                    message = str(exc)
                    logger.warning(f"CodeGraph skipped unreadable path during indexing: {candidate.path} ({message})")
                    counters["errors"] += 1
                    counters["files_skipped"] += 1
                    errors.append(f"{candidate.relative_path}: {message}")
                    repository.upsert_file_and_replace_graph(
                        path=candidate.relative_path,
                        language=candidate.language_id,
                        size=candidate.size,
                        content_hash="",
                        modified_at=candidate.modified_at,
                        status="extraction_failed",
                        errors=(message,),
                        node_count=0,
                        nodes=(),
                        edges=(),
                        unresolved_refs=(),
                    )
                    continue

                if content.is_binary:
                    counters["files_skipped"] += 1
                    continue

                if has_extractor:
                    extraction = self._extract(workspace_root, candidate, workspace_key, content.source or b"")
                else:
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
                    content_hash=content.content_hash,
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

            if status == "complete":
                resolution_limit = _resolution_item_limit(
                    max_files=limit,
                    foreground_max_bytes=self._settings.foreground_max_bytes,
                )
                resolution = CodeGraphReferenceResolver(repository).resolve(
                    source_file_paths=indexed_paths,
                    max_import_nodes=resolution_limit,
                    max_refs=resolution_limit,
                    deadline_monotonic=start + self._settings.max_index_seconds,
                    monotonic=self._monotonic,
                )
                counters["cross_file_calls_resolved"] = resolution.resolved_calls
                counters["cross_file_imports_resolved"] = resolution.resolved_imports
                counters["stale_reference_resolutions_cleared"] = resolution.stale_resolutions_cleared
                counters["cross_file_resolution_truncated"] = int(resolution.truncated)
                counters["cross_file_import_nodes_scanned"] = resolution.import_nodes_scanned
                counters["cross_file_references_scanned"] = resolution.references_scanned

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
                    continue

                has_extractor = language.language_id in self._extractors
                if language.stage == "foundation" and (not language.symbol_extraction or not has_extractor):
                    counters["dependency_missing_language_skipped"] += 1
                    counters["files_skipped"] += 1
                    continue

                candidate = _Candidate(
                    path=resolved,
                    relative_path=relative_path,
                    language_id=language.language_id,
                    stage=language.stage,
                    symbol_extraction=language.symbol_extraction,
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

    def _extract(self, workspace_root: Path, candidate: _Candidate, workspace_key: str, source: bytes) -> ExtractionResult:
        """Run the language extractor for a candidate when one is implemented."""
        extractor = self._extractors.get(candidate.language_id)
        if extractor is None:
            return ExtractionResult()
        try:
            kwargs = {
                "workspace_key": workspace_key,
                "file_path": candidate.relative_path,
                "source": source,
            }
            if candidate.language_id in {"javascript", "typescript"}:
                kwargs["workspace_root"] = workspace_root
            return extractor.extract(**kwargs)
        except (ValueError, OSError, ImportError, AttributeError, TypeError, RuntimeError) as exc:
            logger.opt(exception=exc).warning(f"CodeGraph extractor failed for {candidate.relative_path}: {exc}")
            return ExtractionResult(errors=(str(exc),))

    @staticmethod
    def _read_candidate_content(candidate: _Candidate, *, needs_source: bool) -> _CandidateContent:
        """Read or hash one candidate from a single open stream."""
        digest = hashlib.sha256()
        with candidate.path.open("rb") as handle:
            probe = handle.read(4096)
            if CodeGraphIndexer._is_binary(probe):
                return _CandidateContent(content_hash="", is_binary=True)

            if needs_source:
                source = probe + handle.read()
                return _CandidateContent(
                    content_hash=CodeGraphIndexer._hash_bytes(source),
                    source=source,
                )

            digest.update(probe)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return _CandidateContent(content_hash=digest.hexdigest())

    @staticmethod
    def _hash_bytes(source: bytes) -> str:
        """Return a SHA-256 content hash for indexed file bytes."""
        return hashlib.sha256(source).hexdigest()

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
        "dependency_missing_language_skipped": 0,
        "unsupported_language": 0,
        "errors": 0,
    }


def _resolution_item_limit(*, max_files: int, foreground_max_bytes: int) -> int:
    """Derive an explicit foreground cap for resolver import/ref rows."""
    file_scaled_limit = max(1, int(max_files)) * 1000
    byte_scaled_limit = max(1, int(foreground_max_bytes) // 16)
    return max(1, min(file_scaled_limit, byte_scaled_limit))
