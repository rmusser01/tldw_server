from enum import StrEnum


class RecordKind(StrEnum):
    IDENTITY = "identity"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    CORRECTION = "correction"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    CONVENTION = "convention"
    WORKING_CONTEXT = "working_context"
    LEGACY_UNCLASSIFIED = "legacy_unclassified"


class RecordState(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ScopeKind(StrEnum):
    GLOBAL = "global"
    WORKSPACE = "workspace"


class SyncMode(StrEnum):
    DEVICE_ONLY = "device_only"
    SYNCABLE = "syncable"


class AgentVisibility(StrEnum):
    AGENT_VISIBLE = "agent_visible"
    USER_ONLY = "user_only"


class ProposalState(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class ProposalOperation(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    ARCHIVE = "archive"
    PROMOTE = "promote"


class ToolOperation(StrEnum):
    SEARCH = "search"
    GET = "get"
    PROPOSE = "propose"
    UPDATE = "update"
    PROMOTE = "promote"


class ToolResultStatus(StrEnum):
    APPLIED = "applied"
    PROPOSAL_CREATED = "proposal_created"
    REVIEW_REQUIRED = "review_required"
    PERMISSION_DENIED = "permission_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONFLICT = "conflict"
    PROFILE_LOCKED = "profile_locked"
