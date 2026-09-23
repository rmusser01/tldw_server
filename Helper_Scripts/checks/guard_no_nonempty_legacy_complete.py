#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def main() -> int:
    """Reject non-empty test payloads only at the legacy complete endpoint."""
    repo_root = Path(__file__).resolve().parents[2]
    tests_root = repo_root / "tldw_Server_API" / "tests"
    if not tests_root.exists():
        # If tests folder not present, do nothing
        return 0

    offenders = []
    pattern_url = re.compile(r"/api/v1/chats/[^\"\'\n]+?/complete(?=[/?#\"\'])")
    for py in tests_root.rglob("*.py"):
        if py.name == "test_legacy_complete_deprecation.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if "/api/v1/chats/" not in text or "/complete" not in text:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if ".post(" not in line:
                continue
            if not pattern_url.search(line):
                continue
            window = "\n".join(lines[i : i + 10])
            # Allow empty JSON bodies for legacy endpoint
            if "json={}" in window or "json = {}" in window:
                continue
            # Flag any usage that appears to send a non-empty JSON dict
            if "json={" in window or "json = {" in window:
                offenders.append(f"{py}:{i+1}: {line.strip()}")

    if offenders:
        sys.stderr.write(
            "Found tests posting non-empty JSON bodies to legacy /complete endpoint.\n\n"
        )
        sys.stderr.write("\n".join(offenders) + "\n")
        sys.stderr.write(
            "Use json={} or migrate to /{chat_id}/complete-v2 or /{chat_id}/completions.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
