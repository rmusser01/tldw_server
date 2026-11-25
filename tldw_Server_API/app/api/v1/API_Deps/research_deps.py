"""Dependencies shared by the native and Workspace research routes."""

from tldw_Server_API.app.core.Research.service import ResearchService


def get_research_service() -> ResearchService:
    """Return the default deep research service."""
    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)
