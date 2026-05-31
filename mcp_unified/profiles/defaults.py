"""Default profile-assignment helpers for MCP gateway profile resolution."""

from __future__ import annotations

from collections.abc import Iterable

from mcp_unified.interfaces.storage import ProfileAssignmentStore
from mcp_unified.storage.models import ProfileAssignment

GATEWAY_DEFAULT_ASSIGNMENT_ID = "gateway-default"


def select_gateway_default_assignment(
    assignments: Iterable[ProfileAssignment],
) -> ProfileAssignment | None:
    """Return the effective enabled gateway default assignment, if configured."""
    enabled_defaults = [
        assignment
        for assignment in assignments
        if assignment.is_default
        and assignment.enabled
        and assignment.principal_id is None
        and assignment.workspace_id is None
    ]
    if not enabled_defaults:
        return None
    max_updated_at = max(assignment.updated_at for assignment in enabled_defaults)
    newest = [assignment for assignment in enabled_defaults if assignment.updated_at == max_updated_at]
    return sorted(newest, key=lambda assignment: assignment.id)[0].model_copy(deep=True)


async def load_gateway_default_assignment(
    assignment_store: ProfileAssignmentStore,
) -> ProfileAssignment | None:
    """Load the effective gateway default assignment from an assignment store."""
    assignments = await assignment_store.list_assignments()
    return select_gateway_default_assignment(assignments)
