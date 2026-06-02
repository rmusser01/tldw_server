"""Focused repositories extracted from Media DB."""

from tldw_Server_API.app.core.DB_Management.media_db.repositories.chunks_repository import (
    ChunksRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_versions_repository import (
    DocumentVersionsRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.keywords_repository import (
    KeywordsRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import (
    MediaRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_files_repository import (
    MediaFilesRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_lookup_repository import (
    MediaLookupRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import (
    MediaSearchRepository,
)

__all__ = [
    "ChunksRepository",
    "DocumentVersionsRepository",
    "DocumentWorkspaceRepository",
    "KeywordsRepository",
    "MediaFilesRepository",
    "MediaLookupRepository",
    "MediaSearchRepository",
    "MediaRepository",
]
