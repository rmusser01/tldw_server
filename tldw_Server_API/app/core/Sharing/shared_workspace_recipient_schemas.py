"""Core-owned shared-workspace clone models, re-exported by the API schemas package (TASK-13950.15)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from tldw_Server_API.app.core.Workspaces.workspace_schemas import (
    WorkspaceOperationBase,
)

ShortCode = Annotated[str, StringConstraints(min_length=1, max_length=128)]
CloneWarningCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]


class _RecipientModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SharedWorkspaceCloneProgress(_RecipientModel):
    phase: Literal[
        "queued",
        "authorizing",
        "preparing",
        "sources",
        "notes",
        "artifacts",
        "finalizing",
    ]
    percent: int = Field(ge=0, le=100)
    message_code: CloneWarningCode


class SharedWorkspaceCloneCounts(_RecipientModel):
    sources_attempted: int = Field(ge=0, le=1_000_000_000)
    sources_copied: int = Field(ge=0, le=1_000_000_000)
    sources_failed: int = Field(ge=0, le=1_000_000_000)
    notes_attempted: int = Field(ge=0, le=1_000_000_000)
    notes_copied: int = Field(ge=0, le=1_000_000_000)
    notes_failed: int = Field(ge=0, le=1_000_000_000)
    artifacts_attempted: int = Field(ge=0, le=1_000_000_000)
    artifacts_copied: int = Field(ge=0, le=1_000_000_000)
    artifacts_failed: int = Field(ge=0, le=1_000_000_000)
    media_attempted: int = Field(ge=0, le=1_000_000_000)
    media_copied: int = Field(ge=0, le=1_000_000_000)
    media_failed: int = Field(ge=0, le=1_000_000_000)
    operation_owned_media_count: int = Field(ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def _counts_are_consistent(self):
        for item_name in ("sources", "notes", "artifacts", "media"):
            attempted = getattr(self, f"{item_name}_attempted")
            copied = getattr(self, f"{item_name}_copied")
            failed = getattr(self, f"{item_name}_failed")
            if copied + failed > attempted:
                raise ValueError(
                    f"{item_name} copied plus failed cannot exceed attempted"
                )
        if self.operation_owned_media_count > self.media_copied:
            raise ValueError(
                "operation_owned_media_count cannot exceed media_copied"
            )
        return self


class SharedWorkspaceCloneReadiness(_RecipientModel):
    text_search: Literal["ready", "unavailable"]
    citations: Literal["ready", "unavailable"]
    vector_search: Literal["ready", "needs_indexing", "not_configured"]


class SharedWorkspaceCloneWarning(_RecipientModel):
    code: CloneWarningCode
    count: int = Field(ge=0, le=1_000_000_000)


class SharedWorkspaceCloneResult(_RecipientModel):
    schema_version: Literal[1] = 1
    outcome: Literal["complete", "partial"]
    workspace_id: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    publication_confirmed: bool
    counts: SharedWorkspaceCloneCounts
    readiness: SharedWorkspaceCloneReadiness
    warnings: list[SharedWorkspaceCloneWarning] = Field(
        default_factory=list,
        max_length=8,
    )


class SharedWorkspaceCloneError(_RecipientModel):
    code: ShortCode
    message_key: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=320)
    cleanup_state: Literal["complete", "pending", "unknown"]


class SharedWorkspaceCloneOperationResponse(WorkspaceOperationBase):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    command: Literal["shared_workspace_clone"] = "shared_workspace_clone"
    status: Literal["queued", "running", "succeeded", "failed"]
    share_id: int = Field(gt=0)
    progress: SharedWorkspaceCloneProgress | None = None
    result: SharedWorkspaceCloneResult | None = None
    error: SharedWorkspaceCloneError | None = None

    @model_validator(mode="after")
    def _terminal_shape_matches_status(self):
        if self.status in {"queued", "running"}:
            if self.progress is None or self.result is not None or self.error is not None:
                raise ValueError(
                    f"{self.status} clone operations require progress only"
                )
        elif self.status == "succeeded":
            if (
                self.progress is not None
                or self.result is None
                or self.error is not None
                or not self.result.publication_confirmed
            ):
                raise ValueError(
                    "succeeded clone operations require a confirmed result only"
                )
        elif self.progress is not None or self.result is not None or self.error is None:
            raise ValueError("failed clone operations require an error only")
        return self
