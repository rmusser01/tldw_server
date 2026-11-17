"""Compact card reads for the workspace artifact list."""

import json
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

# The v1 client stores card identity beside its full evidence. Read just the two
# scalars when visibility, schema and types match the client codec.
_SQLITE_IDENTITY = """
      CASE WHEN json_valid(version_metadata_json) THEN
        CASE WHEN json_type(version_metadata_json, '$.workspaceClient.schemaVersion') IN ('integer', 'real')
                  AND json_extract(version_metadata_json, '$.workspaceClient.schemaVersion') = 1
                  AND COALESCE(json_type(version_metadata_json, '$.workspaceClient.removedFromStudio'), 'false') = 'false'
                  AND COALESCE(json_type(version_metadata_json, '$.workspaceClient.id'), 'text') = 'text'
                  AND COALESCE(json_type(version_metadata_json, '$.workspaceClient.version'), 'integer') = 'integer'
             THEN json_array(substr(json_extract(version_metadata_json, '$.workspaceClient.id'), 1, 129),
                             json_extract(version_metadata_json, '$.workspaceClient.version'))
        END
      END"""

_POSTGRES_IDENTITY = """
    CASE WHEN jsonb_path_exists(version_metadata_json::jsonb,
                  '$.workspaceClient ? (@.type() == "object" && @.schemaVersion.type() == "number"
                    && @.schemaVersion == 1 && (!exists(@.id) || @.id.type() == "string")
                    && (!exists(@.version) || @.version.type() == "number")
                    && (!exists(@.removedFromStudio) || (@.removedFromStudio.type() == "boolean"
                      && @.removedFromStudio == false)))')
         THEN json_build_array(left(version_metadata_json::json #>> '{workspaceClient,id}', 129),
                               version_metadata_json::json #> '{workspaceClient,version}')::text
    END"""


def _project_card(row: Any) -> dict[str, Any]:
    """Keep display labels separate from the destination's persistent identity and revision."""
    card = dict(row)
    identity, producer_json = card.pop("identity"), card.pop("producer")
    card.update(display_id=card["id"], display_version=card["version"], defer_detail=False,
                imported=False, model=None, provider=None, created_at=str(card["created_at"]))
    if not identity or not producer_json:
        return card
    try:
        producer = json.loads(producer_json)
    except (ValueError, RecursionError):
        return card
    if not isinstance(producer, dict):
        return card
    imported = producer.get("origin") in ("research_pack_import", "workspace_snapshot_import")
    if imported:
        producer = producer.get("importedProducer")
    if not isinstance(producer, dict) or producer.get("origin") != "research_workspace_output_job":
        return card
    display_id, logical_version = json.loads(identity)
    display_id = card["id"] if display_id is None else display_id
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", display_id):
        return card
    if logical_version is not None and (type(logical_version) is not int or not 0 <= logical_version < 2**53):
        return card
    settings = producer.get("settings")
    for key, limit in (("model", 256), ("provider", 128)):
        value = settings.get(key) if isinstance(settings, dict) else None
        if isinstance(value, str):
            if len(value) > limit:
                return card
            card[key] = value.strip() or None
    card.update(display_id=display_id, defer_detail=True, imported=imported)
    if imported and logical_version is not None:
        card["display_version"] = logical_version
    return card


def list_workspace_artifact_revisions(db: CharactersRAGDB, workspace_id: str) -> list[dict[str, Any]]:
    """Return card columns and validated display identities without loading artifact bodies into the API."""
    identity = _POSTGRES_IDENTITY if db.backend_type == BackendType.POSTGRESQL else _SQLITE_IDENTITY
    query = f"""SELECT id, workspace_id, version, title, artifact_type, status, review_state, created_at,
                       CASE WHEN length(producer_metadata_json) <= 65536 THEN producer_metadata_json END AS producer,
                       CASE WHEN artifact_type IN ('summary', 'compare_sources', 'report')
                                 AND status IN ('complete', 'completed', 'failed') AND review_state <> 'archived'
                            THEN {identity} END AS identity
                FROM workspace_artifacts WHERE workspace_id = ? ORDER BY created_at DESC"""  # nosec B608 - fixed SQL fragments, bound workspace ID
    try:
        rows = db.execute_query(query, (workspace_id,), read_only=True).fetchall()
    except CharactersRAGDBError:
        if db.backend_type != BackendType.POSTGRESQL:
            raise
        # Legacy TEXT accepts JSON that PostgreSQL JSONB cannot represent, including escaped NULs.
        # Retain all rows through the original full-detail reader in that case.
        logger.warning("Workspace artifact JSON projection unavailable; loading complete artifact details")
        rows = db.execute_query(
            """SELECT id, workspace_id, version, title, artifact_type, status, review_state, created_at,
                      NULL AS producer, NULL AS identity
               FROM workspace_artifacts WHERE workspace_id = ? ORDER BY created_at DESC""",
            (workspace_id,), read_only=True,
        ).fetchall()
    return [_project_card(row) for row in rows]
