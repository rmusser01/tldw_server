#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import sys


PROTECTED_PREFIXES = (
    "admin-ui/",
    "apps/tldw-frontend/",
    "apps/extension/",
    "apps/packages/ui/",
)
GOVERNANCE_PREFIXES = ("LICENSES/",)
GOVERNANCE_PATHS = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.txt",
    "Helper_Scripts/ci/check_frontend_license_gate.py",
    ".github/workflows/frontend-required.yml",
}
CONTRACT_PREFIXES = ("tldw_Server_API/app/api/v1/",)
CONTRACT_PATHS = {"tldw_Server_API/app/main.py"}


def _matches(path: str) -> bool:
    return (
        path in GOVERNANCE_PATHS
        or path in CONTRACT_PATHS
        or path.startswith(PROTECTED_PREFIXES)
        or path.startswith(GOVERNANCE_PREFIXES)
        or path.startswith(CONTRACT_PREFIXES)
    )


def blocked_changes(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if path and _matches(path)]


def evaluate(*, author: str, owner: str, paths: Iterable[str]) -> list[str]:
    if author.casefold() == owner.casefold():
        return []
    return blocked_changes(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args(argv)
    blocked = evaluate(
        author=args.author,
        owner=args.owner,
        paths=(line.strip() for line in sys.stdin),
    )
    if not blocked:
        return 0
    print(
        "Temporary licensing gate: this PR author cannot modify these paths "
        "until the required contributor grants are active:",
        file=sys.stderr,
    )
    for path in blocked:
        print(f"- {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
