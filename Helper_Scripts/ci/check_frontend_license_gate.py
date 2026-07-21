from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from typing import BinaryIO

MAX_INPUT_BYTES = 8 * 1024 * 1024

PROTECTED_PREFIXES = (
    "admin-ui/",
    "apps/tldw-frontend/",
    "apps/extension/",
    "apps/packages/ui/",
    "LICENSES/",
    "tldw_Server_API/app/api/v1/",
)

PROTECTED_EXACT = frozenset(
    {
        "LICENSE",
        "THIRD_PARTY_NOTICES.txt",
        "Helper_Scripts/ci/check_frontend_license_gate.py",
        ".github/workflows/frontend-license-gate.yml",
        ".github/workflows/frontend-required.yml",
        "tldw_Server_API/app/main.py",
    }
)


def blocked_changes(paths: Iterable[str]) -> list[str]:
    return [
        path
        for path in paths
        if path in PROTECTED_EXACT or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    ]


def evaluate(*, author: str, owner: str, paths: Iterable[str]) -> list[str]:
    if author.casefold() == owner.casefold():
        return []
    return blocked_changes(paths)


def read_nul_paths(
    stream: BinaryIO,
    *,
    max_bytes: int = MAX_INPUT_BYTES,
) -> list[str]:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ValueError(f"changed-path input exceeds {max_bytes} bytes")
    return [value.decode("utf-8", errors="surrogateescape") for value in data.split(b"\0") if value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--null", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = read_nul_paths(sys.stdin.buffer)
    except (OSError, ValueError) as exc:
        print(f"frontend license gate input error: {exc}", file=sys.stderr)
        return 2

    blocked = evaluate(author=args.author, owner=args.owner, paths=paths)
    if not blocked:
        print("frontend license gate: allowed")
        return 0

    print("frontend license gate: protected changes are frozen", file=sys.stderr)
    for path in blocked:
        print(f"- {ascii(path)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
