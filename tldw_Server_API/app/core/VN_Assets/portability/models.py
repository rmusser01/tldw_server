"""Lightweight data models for VN pack portability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VNPackExportOptions:
    include_character_payload: bool = False
    include_world_book_payloads: bool = False
    include_full_provenance: bool = False
    strict: bool = False
    warn_for_sharing: bool = True


@dataclass(frozen=True)
class VNPackExportResult:
    archive_path: Path
    archive_sha256: str
    canonical_payload_fingerprint: str
    file_size_bytes: int
    warnings: list[str]
