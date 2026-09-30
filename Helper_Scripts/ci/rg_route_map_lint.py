"""Fail CI when a Resource Governor route_map entry is dead, shadowed or unused.

Builds the fully enabled app the same way the route-auth ratchet does. The
allowlist holds intentional exceptions, one problem string per line, with a
``#`` comment giving the reason.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).resolve().parent / "rg_route_map_lint_allowlist.txt"


def lint(route_map: Mapping[str, Any], served: list[Any], allow: set[str]) -> list[str]:
    from tldw_Server_API.app.core.Resource_Governance.policy_resolver import compile_route_glob

    paths = [r.path for r in served if getattr(r, "path", None) and getattr(r, "methods", None)]
    patterns = [(str(p), compile_route_glob(str(p))) for p in (route_map.get("by_path") or {})]
    problems: list[str] = []
    for i, (raw, rx) in enumerate(patterns):
        hits = [p for p in paths if rx.match(p)]
        if not hits:
            problems.append(f"by_path {raw} matches no served route")
        elif all(any(earlier.match(p) for _r, earlier in patterns[:i]) for p in hits):
            problems.append(f"by_path {raw} is shadowed by earlier patterns")
    used_tags = {t for r in served for t in (getattr(r, "tags", None) or ())}
    for tag in route_map.get("by_tag") or {}:
        if str(tag) not in used_tags:
            problems.append(f"by_tag {tag} is used by no served route")
    return [p for p in problems if p not in allow]


def _allowlist() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    out = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import asyncio

    from Helper_Scripts.ci.route_auth_ratchet import load_app
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import default_policy_loader
    from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

    app = load_app()
    loader = default_policy_loader()
    asyncio.run(loader.load_once())
    problems = lint(loader.get_snapshot().route_map or {}, list(iter_served_routes(app.routes)), _allowlist())
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
