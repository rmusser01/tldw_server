"""Ratchet: SSE data/DONE frames are built with core/LLM_Calls/sse.py helpers.

Hand-rolled ``f"data: {json.dumps(...)}\\n\\n"`` and ``"data: [DONE]\\n\\n"`` literals
drift (TASK-13332); use ``sse_data`` / ``sse_event`` / ``sse_done`` instead.
The baseline lists the sites that are intentionally not the helper's bytes.
Lower a count when you remove a site; never raise one.
"""

from __future__ import annotations

import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

INLINE_FRAME = re.compile(
    r"""f["']data: \{_?json\.dumps\(|["']data: \[DONE\]\\n\\n["']"""
)

BASELINE = {
    # The helpers themselves.
    "core/LLM_Calls/sse.py": 2,
    # Anthropic wire format: event frame with explicit ensure_ascii.
    "core/LLM_Calls/anthropic_messages.py": 1,
    # Unterminated line handed to SSEStream.send_raw_sse_line.
    "api/v1/endpoints/character_chat_sessions.py": 1,
    # Compact separators=(",", ":") replay frames.
    "api/v1/endpoints/chat.py": 2,
}


def _inline_frame_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in APP_ROOT.rglob("*.py"):
        rel = path.relative_to(APP_ROOT).as_posix()
        if "/tests/" in f"/{rel}":
            continue
        hits = len(INLINE_FRAME.findall(path.read_text(encoding="utf-8", errors="ignore")))
        if hits:
            counts[rel] = hits
    return counts


def test_no_new_inline_sse_frames() -> None:
    over = {
        rel: (count, BASELINE.get(rel, 0))
        for rel, count in _inline_frame_counts().items()
        if count > BASELINE.get(rel, 0)
    }
    assert not over, (
        "Inline SSE frame construction found; use sse_data/sse_event/sse_done from "
        "tldw_Server_API.app.core.LLM_Calls.sse. (file: (found, allowed)) " + repr(over)
    )


def test_inline_sse_frame_baseline_is_tight() -> None:
    counts = _inline_frame_counts()
    stale = {rel: (counts.get(rel, 0), allowed) for rel, allowed in BASELINE.items() if counts.get(rel, 0) < allowed}
    assert not stale, "Lower the BASELINE for sites that were removed. (file: (found, allowed)) " + repr(stale)
