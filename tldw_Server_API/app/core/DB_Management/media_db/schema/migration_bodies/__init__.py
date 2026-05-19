"""Package-native Postgres migration body helpers."""

from __future__ import annotations

from .postgres_collections import (
    PostgresCollectionsBody,
    run_postgres_migrate_to_v12,
    run_postgres_migrate_to_v13,
)
from .postgres_early_schema import (
    PostgresEarlySchemaBody,
    run_postgres_migrate_to_v5,
    run_postgres_migrate_to_v6,
    run_postgres_migrate_to_v7,
    run_postgres_migrate_to_v8,
)
from .postgres_data_tables import (
    PostgresDataTablesBody,
    run_postgres_migrate_to_v14,
    run_postgres_migrate_to_v15,
)
from .postgres_email_schema import (
    PostgresEmailSchemaBody,
    run_postgres_migrate_to_v22,
)
from .postgres_transcript_run_history import (
    PostgresTranscriptRunHistoryBody,
    run_postgres_migrate_to_v23,
)
from .postgres_source_hash import (
    PostgresSourceHashBody,
    run_postgres_migrate_to_v16,
)
from .postgres_sequence_sync import (
    PostgresSequenceSyncBody,
    run_postgres_migrate_to_v18,
)
from .postgres_structure_visual_indexes import (
    PostgresStructureVisualIndexBody,
    run_postgres_migrate_to_v21,
)
from .postgres_tts_history import (
    PostgresTTSHistoryBody,
    run_postgres_migrate_to_v20,
)
from .postgres_fts_rls import (
    PostgresFTSRLSBody,
    run_postgres_migrate_to_v19,
)
from .postgres_visibility_owner import (
    PostgresVisibilityOwnerBody,
    run_postgres_migrate_to_v9,
)
from .postgres_claims import (
    PostgresClaimsBody,
    run_postgres_migrate_to_v10,
    run_postgres_migrate_to_v17,
)
from .postgres_mediafiles import (
    PostgresMediaFilesBody,
    run_postgres_migrate_to_v11,
)

__all__ = [
    "PostgresVisibilityOwnerBody",
    "run_postgres_migrate_to_v9",
    "PostgresClaimsBody",
    "run_postgres_migrate_to_v10",
    "PostgresMediaFilesBody",
    "run_postgres_migrate_to_v11",
    "run_postgres_migrate_to_v17",
    "PostgresEarlySchemaBody",
    "run_postgres_migrate_to_v5",
    "run_postgres_migrate_to_v6",
    "run_postgres_migrate_to_v7",
    "run_postgres_migrate_to_v8",
    "PostgresCollectionsBody",
    "run_postgres_migrate_to_v12",
    "run_postgres_migrate_to_v13",
    "PostgresDataTablesBody",
    "run_postgres_migrate_to_v14",
    "run_postgres_migrate_to_v15",
    "PostgresEmailSchemaBody",
    "run_postgres_migrate_to_v22",
    "PostgresTranscriptRunHistoryBody",
    "run_postgres_migrate_to_v23",
    "PostgresSourceHashBody",
    "run_postgres_migrate_to_v16",
    "PostgresSequenceSyncBody",
    "run_postgres_migrate_to_v18",
    "PostgresStructureVisualIndexBody",
    "run_postgres_migrate_to_v21",
    "PostgresTTSHistoryBody",
    "run_postgres_migrate_to_v20",
    "PostgresFTSRLSBody",
    "run_postgres_migrate_to_v19",
]
