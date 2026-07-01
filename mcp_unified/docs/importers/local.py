from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from ..errors import DocsError
from ..models import AccessScope
from ..settings import DocsSettings
from ..store.sqlite import DocsCatalogStore
from .base import ParsedDocument, chunks_from_text
from .html import parse_html
from .markdown import parse_markdown

SUPPORTED_SUFFIXES = {".htm", ".html", ".markdown", ".md", ".mdx", ".text", ".txt"}


class DocsImportService:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore) -> None:
        self.settings = settings
        self.store = store

    def import_path(
        self,
        *,
        scope: AccessScope,
        path: str | Path,
        keywords: Iterable[str],
        collection_names: Iterable[str],
    ) -> dict:
        target = self._assert_allowed_path(Path(path))
        files = self._iter_import_files(target)
        imported: list[dict] = []

        for file_path in files:
            parsed = self._parse_file(file_path)
            chunk_texts = chunks_from_text(parsed.text)
            chunks = [
                {
                    "text": chunk,
                    "citation": f"{file_path.name}:{idx + 1}",
                }
                for idx, chunk in enumerate(chunk_texts)
            ]
            document_id = self.store.upsert_document(
                scope=scope,
                title=parsed.title,
                document_type=parsed.document_type,
                canonical_uri=parsed.canonical_uri,
                source_path=parsed.source_path,
                source_url=parsed.source_url,
                text=parsed.text,
                sections=[asdict(section) for section in parsed.sections],
                chunks=chunks,
                keywords=keywords,
                collection_names=collection_names,
                metadata={"importer": "local"},
            )
            imported.append({"id": document_id, "title": parsed.title, "chunks": len(chunks)})

        return {"status": "created" if imported else "unchanged", "documents": imported}

    def _assert_allowed_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        for root in self.settings.trusted_roots:
            trusted_root = root.expanduser().resolve()
            try:
                resolved.relative_to(trusted_root)
            except ValueError:
                continue
            return resolved
        raise DocsError(
            code="path_scope_denied",
            message="Path is outside configured trusted roots.",
            details={"path": str(resolved)},
        )

    def _iter_import_files(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target]
        if not target.is_dir():
            raise DocsError(
                code="import_path_not_found",
                message="Import path does not exist.",
                details={"path": str(target)},
            )

        files: list[Path] = []
        for candidate in sorted(target.rglob("*")):
            if not candidate.is_file():
                continue
            files.append(self._assert_allowed_path(candidate))
        return files

    def _parse_file(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise DocsError(
                code="unsupported_import_format",
                message="Unsupported local import file type.",
                details={"path": str(path), "suffix": suffix},
            )
        self._assert_file_size(path)
        text = path.read_text(encoding="utf-8")

        if suffix in {".md", ".markdown"}:
            return parse_markdown(path, text, "markdown")
        if suffix == ".mdx":
            return parse_markdown(path, text, "mdx")
        if suffix in {".txt", ".text"}:
            return parse_markdown(path, text, "text")
        if suffix in {".html", ".htm"}:
            return parse_html(path, text)

        raise DocsError(
            code="unsupported_import_format",
            message="Unsupported local import file type.",
            details={"path": str(path), "suffix": suffix},
        )

    def _assert_file_size(self, path: Path) -> None:
        file_size = path.stat().st_size
        if file_size > self.settings.max_import_file_bytes:
            raise DocsError(
                code="import_file_too_large",
                message="Import file exceeds configured maximum size.",
                details={
                    "path": str(path),
                    "size": file_size,
                    "max_import_file_bytes": self.settings.max_import_file_bytes,
                },
            )
