"""Package-owned helper for safe-metadata writes (moved from core/Utils/metadata_utils.py, TASK-13950.12)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


def update_version_safe_metadata_in_transaction(
    db: Any,
    dv_id: int,
    safe_metadata_json: str | None,
    merged_metadata: Mapping[str, Any],
    connection: Any,
) -> None:
    """
    Update DocumentVersions.safe_metadata (and bump version/last_modified)
    for a specific version and keep the identifier index in sync.

    This helper encapsulates the repeated SQL + identifier upsert logic used
    by the safe-metadata write endpoints.
    """
    try:
        now_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except Exception:  # pragma: no cover - extremely unlikely
        now_ts = None

    if now_ts is not None:
        db.execute_query(
            "UPDATE DocumentVersions SET safe_metadata=?, version=version+1, last_modified=? WHERE id=? AND deleted=0",
            (safe_metadata_json, now_ts, dv_id),
            connection=connection,
        )
    else:
        db.execute_query(
            "UPDATE DocumentVersions SET safe_metadata=?, version=version+1 WHERE id=? AND deleted=0",
            (safe_metadata_json, dv_id),
            connection=connection,
        )

    def _identifier_index_failure_is_compatible(exc: Exception) -> bool:
        """Return whether an identifier index error matches legacy-compatible failures."""
        message = str(exc or "").lower()
        compatible_markers = (
            "no such table",
            "does not exist",
            "undefined table",
            "unsupported upsert",
        )
        return any(marker in message for marker in compatible_markers)

    # Maintain identifier index using backend-specific upsert.
    # Accept both canonical and legacy key variants for robustness.
    try:
        doi = merged_metadata.get("doi") or merged_metadata.get("DOI")
        pmid = merged_metadata.get("pmid") or merged_metadata.get("PMID")
        pmcid = merged_metadata.get("pmcid") or merged_metadata.get("PMCID")
        arxiv = (
            merged_metadata.get("arxiv_id")
            or merged_metadata.get("arxiv")
            or merged_metadata.get("ArXiv")
        )
        s2id = merged_metadata.get("s2_paper_id") or merged_metadata.get("paperId")
        backend = db.backend_type() if callable(db.backend_type) else db.backend_type
        if backend == BackendType.POSTGRESQL:
            ident_sql = (
                "INSERT INTO DocumentVersionIdentifiers (dv_id, doi, pmid, pmcid, arxiv_id, s2_paper_id) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (dv_id) DO UPDATE SET "
                "doi = EXCLUDED.doi, pmid = EXCLUDED.pmid, pmcid = EXCLUDED.pmcid, "
                "arxiv_id = EXCLUDED.arxiv_id, s2_paper_id = EXCLUDED.s2_paper_id"
            )
        else:
            ident_sql = (
                "INSERT OR REPLACE INTO DocumentVersionIdentifiers (dv_id, doi, pmid, pmcid, arxiv_id, s2_paper_id) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
        db.execute_query(
            ident_sql,
            (dv_id, doi, pmid, pmcid, arxiv, s2id),
            connection=connection,
        )
    except (sqlite3.OperationalError, DatabaseError) as exc:
        # Missing identifier table or unsupported upsert is not fatal.
        if not _identifier_index_failure_is_compatible(exc):
            raise
        logger.debug(
            'Identifier index update skipped (missing table/unsupported upsert) for dv_id={}: {}',
            dv_id,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            'Identifier index update failed for dv_id={}: {}',
            dv_id,
            exc,
            exc_info=True,
        )
        raise
