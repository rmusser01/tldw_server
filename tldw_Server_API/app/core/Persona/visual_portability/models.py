"""Lightweight data models for persona visual pack portability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaVisualPackExportOptions:
    strict: bool = False
    include_full_provenance: bool = False
    warn_for_sharing: bool = True


@dataclass(frozen=True)
class PersonaVisualPackExportResult:
    archive_path: Path
    archive_sha256: str
    canonical_payload_fingerprint: str
    file_size_bytes: int
    warnings: list[str]

