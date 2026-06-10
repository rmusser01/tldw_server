"""Service layer for Scheduled Tasks automation definition foundations."""

from __future__ import annotations

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskActionCapability,
    ScheduledTaskAuditListResponse,
    ScheduledTaskAutomationCapabilitiesResponse,
    ScheduledTaskAutomationCapability,
    ScheduledTaskDefinitionListResponse,
    ScheduledTaskPreviewListResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL


class ScheduledTaskAutomationService:
    """Business service for Scheduled Tasks-owned automation definitions."""

    def get_capabilities(self) -> ScheduledTaskAutomationCapabilitiesResponse:
        """Return Phase 4B capabilities without exposing execution support."""
        return ScheduledTaskAutomationCapabilitiesResponse(
            items=[
                ScheduledTaskAutomationCapability(
                    family="recurring_question",
                    family_availability="available",
                    actions=self._definition_actions(),
                    related_capabilities={"rag": {"status": "not_checked"}},
                ),
                ScheduledTaskAutomationCapability(
                    family="agent_task",
                    family_availability="available",
                    actions=self._definition_actions(),
                    related_capabilities={"acp": {"status": "not_checked"}},
                ),
            ]
        )

    def list_previews(self, *, limit: int, offset: int) -> ScheduledTaskPreviewListResponse:
        """Return an empty preview page until durable preview storage lands."""
        return ScheduledTaskPreviewListResponse(limit=limit, offset=offset)

    def list_definitions(self, *, limit: int, offset: int) -> ScheduledTaskDefinitionListResponse:
        """Return an empty definition page until definition storage lands."""
        return ScheduledTaskDefinitionListResponse(limit=limit, offset=offset)

    def list_audit_events(self, *, limit: int, offset: int) -> ScheduledTaskAuditListResponse:
        """Return an empty audit page until definition audit storage lands."""
        return ScheduledTaskAuditListResponse(limit=limit, offset=offset)

    @staticmethod
    def _definition_actions() -> dict[str, ScheduledTaskActionCapability]:
        return {
            "preview": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "create_definition": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "update_definition": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "pause": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "resume": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "archive": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "duplicate": ScheduledTaskActionCapability(
                status="available",
                required_permissions=[TASKS_CONTROL],
            ),
            "execute": ScheduledTaskActionCapability(
                status="unavailable",
                reason="execution_not_implemented",
                required_permissions=[TASKS_CONTROL],
            ),
        }
