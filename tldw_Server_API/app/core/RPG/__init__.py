"""Core RPG runtime primitives."""

from tldw_Server_API.app.core.RPG.authority import AuthorityDecision, decide_authority

__all__ = [
    "AuthorityDecision",
    "RPGProposalRecord",
    "RPGService",
    "RPGServiceProposal",
    "RecordEventsResult",
    "SnapshotResult",
    "decide_authority",
]


def __getattr__(name: str) -> object:
    if name == "RPGProposalRecord":
        from tldw_Server_API.app.core.RPG.proposals import RPGProposalRecord

        return RPGProposalRecord
    if name in {"RPGService", "RPGServiceProposal", "RecordEventsResult", "SnapshotResult"}:
        from tldw_Server_API.app.core.RPG.service import (
            RecordEventsResult,
            RPGService,
            RPGServiceProposal,
            SnapshotResult,
        )

        return {
            "RPGService": RPGService,
            "RPGServiceProposal": RPGServiceProposal,
            "RecordEventsResult": RecordEventsResult,
            "SnapshotResult": SnapshotResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
