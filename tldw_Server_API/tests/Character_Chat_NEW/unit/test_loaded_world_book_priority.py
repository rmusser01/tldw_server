"""Frozen and live lore honor the same priority before budget admission."""

from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
    WorldBookEntry,
    match_world_book_entries,
)


def test_frozen_scan_candidate_keeps_priority_over_quick_index_candidate():
    """A low-priority indexed match must not consume the higher match's budget."""
    low = WorldBookEntry(
        entry_id=1, world_book_id=1, keywords=["harbor"], content="Low", priority=1
    )
    high = WorldBookEntry(
        entry_id=2,
        world_book_id=1,
        keywords=["harbor"],
        content="High",
        priority=10,
        case_sensitive=True,
    )
    response, entries = match_world_book_entries(
        "harbor",
        [{"id": 1}],
        {1: [low, high]},
        token_budget=1,
        recursive_scanning=False,
        count_tokens=lambda _text: 1,
    )
    assert (response["processed_context"], [entry.entry_id for entry in entries]) == (  # nosec B101 - pytest behavior assertion.
        "High",
        [2],
    )
