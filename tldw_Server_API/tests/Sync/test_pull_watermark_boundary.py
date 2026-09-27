"""A pull cursor may never advance past an envelope that was withheld.

ADR-034 makes an unresolved materialization conflict an ordering blocker: envelopes at
or beyond it are withheld, not delivered. The versioned pull path derived its boundary
from the SAFE subset and advanced watermarks only up to it. The legacy adapter-v1 path
took max() over the RAW list, which includes the withheld envelopes -- so it advanced
the cursor past envelopes it had never delivered, and returned has_more=False.

That is silent, permanent, per-device data loss on the default path: a client that never
negotiated supported_adapter_versions resolves to v1, and is told it is caught up.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tldw_Server_API.app.core.Sync.v2.service import _safe_pull_boundary


@dataclass
class _Env:
    server_sequence: int


def test_boundary_never_passes_the_blocker() -> None:
    """The reproduced scenario: seq 1 blocked, seq 2 deliverable, nothing delivered."""
    raw = [_Env(1), _Env(2)]
    boundary = _safe_pull_boundary(
        raw_envelopes=raw,
        page=[],
        has_visible_lookahead=False,
        blocker_cursor=1,
        restore_barrier=None,
        default=0,
    )
    assert boundary == 0, (
        "advancing to 2 would skip seq 2 forever: it was withheld, not delivered"
    )


def test_boundary_uses_the_page_when_there_is_more_visible() -> None:
    page = [_Env(5), _Env(6)]
    assert _safe_pull_boundary(
        raw_envelopes=[_Env(5), _Env(6), _Env(7)],
        page=page,
        has_visible_lookahead=True,
        blocker_cursor=None,
        restore_barrier=None,
        default=0,
    ) == 6


def test_restore_barrier_also_bounds_the_boundary() -> None:
    assert _safe_pull_boundary(
        raw_envelopes=[_Env(1), _Env(2), _Env(3)],
        page=[],
        has_visible_lookahead=False,
        blocker_cursor=None,
        restore_barrier=2,
        default=0,
    ) == 1


def test_a_restore_barrier_overrides_the_page_shortcut() -> None:
    """Matches the versioned path: the page shortcut only applies with no barrier."""
    assert _safe_pull_boundary(
        raw_envelopes=[_Env(1), _Env(2)],
        page=[_Env(1), _Env(2)],
        has_visible_lookahead=True,
        blocker_cursor=None,
        restore_barrier=2,
        default=0,
    ) == 1


def test_unblocked_advances_to_the_last_raw_envelope() -> None:
    assert _safe_pull_boundary(
        raw_envelopes=[_Env(1), _Env(2), _Env(3)],
        page=[_Env(1), _Env(2), _Env(3)],
        has_visible_lookahead=False,
        blocker_cursor=None,
        restore_barrier=None,
        default=0,
    ) == 3


def test_empty_scan_returns_the_default() -> None:
    assert _safe_pull_boundary(
        raw_envelopes=[], page=[], has_visible_lookahead=False,
        blocker_cursor=None, restore_barrier=None, default=42,
    ) == 42


def test_everything_blocked_returns_the_default() -> None:
    """Nothing safe to advance to, so the cursor must not move at all."""
    assert _safe_pull_boundary(
        raw_envelopes=[_Env(7), _Env(8)], page=[], has_visible_lookahead=False,
        blocker_cursor=7, restore_barrier=None, default=6,
    ) == 6


def test_the_liveness_interaction_this_fix_nearly_broke() -> None:
    """Refusing to advance is only half the contract.

    tests/Sync/test_sync_v2_service.py asserts, inside its pagination loop:

        assert page.next_cursor != cursor or not page.has_more

    i.e. a pull must either advance the cursor or declare itself finished. A first
    attempt at this fix stopped the cursor without touching has_more, which turned a
    silent-data-loss bug into a livelock -- the client re-requesting the same cursor
    forever. `pull` therefore clears has_more when the boundary cannot advance and
    nothing was delivered: there is nothing more deliverable until the blocker is
    resolved, and the client re-polls later.

    This test pins the boundary half; the service suite pins the liveness half.
    """
    raw = [_Env(1), _Env(2), _Env(3)]
    boundary = _safe_pull_boundary(
        raw_envelopes=raw,
        page=[],
        has_visible_lookahead=False,
        blocker_cursor=1,
        restore_barrier=None,
        default=0,
    )
    assert boundary == 0
    # The caller must recognise "did not advance" and stop, rather than loop.
    assert boundary <= 0, "pull() must clear has_more when the boundary cannot advance"
