"""Validate the references accepted by a native Workspace research launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


Identifier = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")
]
Revision = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]


class _LaunchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class WorkspaceRunSource(_LaunchModel):
    source_id: Identifier
    catalog_item_id: Identifier
    source_type: Identifier
    association_version: Revision
    catalog_item_version: Revision | None = None
    ready_modes: tuple[Literal["fts", "semantic", "hybrid"], ...] = Field(default=(), max_length=3)


class WorkspaceRunOutput(_LaunchModel):
    owner_kind: Literal["server_artifact"]
    owner_id: Identifier
    owner_version: Revision


class WorkspaceRunLaunch(_LaunchModel):
    launch_id: Identifier
    query: str = Field(min_length=1, max_length=4000)
    created_at: AwareDatetime
    source_snapshot: tuple[WorkspaceRunSource, ...] = Field(min_length=1, max_length=100)
    initiating_output: WorkspaceRunOutput | None = None
    initiating_message_id: Identifier | Literal[""] = ""
    server_chat_id: Identifier | Literal[""] = ""

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if not isinstance(value, str) or "\x00" in value or len(value) > 4000:
            raise ValueError("Invalid research question")
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_references(self):
        sources = self.source_snapshot
        if (
            len({row.source_id for row in sources}) != len(sources)
            or len({row.catalog_item_id for row in sources}) != len(sources)
            or any(len(set(row.ready_modes)) != len(row.ready_modes) for row in sources)
            or not any(row.ready_modes for row in sources)
        ):
            raise ValueError("Select distinct sources with at least one source ready")
        if self.initiating_message_id and (not self.server_chat_id or self.initiating_output):
            raise ValueError("The initiating message requires its own workspace conversation")
        return self


class WorkspaceRunOrigin(_LaunchModel):
    workspace_id: Identifier
    owner_user_id: Identifier
    launch: WorkspaceRunLaunch


class WorkspaceRunCapability(_LaunchModel):
    contract: Literal["research-workspace-runs.v1"] = "research-workspace-runs.v1"


class WorkspaceRunOriginConflict(ValueError):
    """The captured Workspace references changed or are no longer available."""


def require_workspace(db: CharactersRAGDB, workspace_id: str, owner_user_id: str, *, conn=None) -> dict:
    if db.owner_user_id != owner_user_id:
        raise KeyError(workspace_id)
    workspace = db.workspace_store.get_workspace(workspace_id, conn=conn, for_update=conn is not None)
    if workspace is None:
        raise KeyError(workspace_id)
    return workspace


def validate_workspace_launch(db: CharactersRAGDB, origin: WorkspaceRunOrigin) -> None:
    """Check the current selected associations and initiating owner under workspace admission."""
    workspace_id, launch = origin.workspace_id, origin.launch
    with db.transaction() as conn:
        workspace = require_workspace(db, workspace_id, origin.owner_user_id, conn=conn)
        if workspace.get("archived") or workspace.get("native_chat_admission_closed"):
            raise WorkspaceRunOriginConflict("Workspace is closed to new research runs")
        for captured in launch.source_snapshot:
            source = db.workspace_store.get_workspace_source(workspace_id, captured.source_id)
            if (
                source is None
                or not source["selected"]
                or str(source["media_id"]) != captured.catalog_item_id
                or (source.get("source_type") or "media") != captured.source_type
                or source["version"] != captured.association_version
            ):
                raise WorkspaceRunOriginConflict("Selected workspace sources changed")
        if output := launch.initiating_output:
            artifact = db.workspace_store.get_workspace_artifact(workspace_id, output.owner_id)
            if artifact is None or artifact["version"] != output.owner_version:
                raise WorkspaceRunOriginConflict("The initiating workspace artifact changed")
        if launch.server_chat_id:
            chat = db.conversation_store.get_conversation_by_id(launch.server_chat_id)
            if (
                chat is None
                or chat.get("scope_type") != "workspace"
                or chat.get("workspace_id") != workspace_id
                or str(chat.get("client_id")) != origin.owner_user_id
            ):
                raise WorkspaceRunOriginConflict("The initiating conversation is unavailable")
            message_id = launch.initiating_message_id
            if (
                message_id
                and db.message_store.get_message_conversation_id(message_id) != launch.server_chat_id
            ):
                raise WorkspaceRunOriginConflict("The initiating message is unavailable")
