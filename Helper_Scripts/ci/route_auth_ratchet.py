#!/usr/bin/env python3
"""Fail when a route is added without an authentication dependency.

There is no global auth middleware in this app, so a route that declares no
auth dependency is simply public.  Three such routes shipped before anyone
noticed (``connectors.py`` returned any user's decrypted job result to an
unauthenticated caller), and nothing in CI would have caught a fourth.

This ratchet walks the *built* application rather than the source, because
authentication is frequently applied at ``include_router(dependencies=[...])``
time -- ``admin/__init__.py`` guards 32 sub-routers that way, and a source scan
would report every one of them as unprotected.

Coverage is deliberately coarse: a route counts as protected when any callable
in its dependency tree is a known authenticator.  Rate limiters do not count,
however they are named -- ``rbac_rate_limit("sharing.admin")`` reads like an
RBAC gate at the call site and is only a requests-per-minute budget, which is
exactly how three ``/sharing/admin/*`` routes ended up unguarded.

Routes that are public on purpose (login, health, HMAC webhooks) live in the
baseline.  The list may shrink, never grow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path(__file__).resolve().parent / "route_auth_baseline.txt"

# Route families are gated by policy, so a default import hides whole modules --
# including ``connectors``, which carried the original defect.  Force them on so
# the gate sees every route the codebase can serve.
ROUTE_POLICY_ENV = {
    "ROUTES_STABLE_ONLY": "false",
    "ROUTES_ENABLE": ",".join(
        [
            "audiobooks",
            "benchmarks",
            "companion",
            "connectors",
            "discord",
            "guardian",
            "integrations",
            "mcp",
            "meetings",
            "personalization",
            "sandbox",
            "self-monitoring",
            "slack",
            "telegram",
        ]
    ),
}

# Callables that establish who the caller is.  A route reaching any of these has
# been authenticated; what it then does with the identity is a different gate's
# problem.
AUTHENTICATORS = frozenset(
    {
        "get_auth_principal",
        "get_chat_workflows_user",
        "get_current_active_user",
        "get_current_user",
        "get_eval_request_user",
        "get_mcp_auth_context",
        "get_prompt_studio_user",
        "get_request_user",
        "require_expected_user",
        "require_local_setup_access",
        "require_service_principal",
        "require_shared_audio_installer_access",
        "resolve_user_id_for_request",
        "verify_api_key",
        "verify_jwt_and_fetch_user",
        "verify_prompts_user",
        "_require_first_run_write_access",
        "_require_mcp_tools_catalog_access",
        "_require_mcp_tools_setup_or_admin_access",
        "_require_setup_write_access",
        "_require_system_configure_access",
    }
)

# Dependency factories whose closures authorize.  Matched on the *factory* name
# because the closure itself is called ``_checker``.
AUTHORIZER_FACTORIES = frozenset(
    {
        "require_api_key_scope",
        "require_org_role",
        "require_permissions",
        "require_project_access",
        "require_roles",
        "require_token_scope",
    }
)


class RatchetError(RuntimeError):
    """Raised when the route inventory cannot be built or read."""


def _load_app():
    for key, value in ROUTE_POLICY_ENV.items():
        os.environ[key] = value
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TEST_MODE", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from tldw_Server_API.app.main import app
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        raise RatchetError(f"could not build the FastAPI app: {exc}") from exc
    return app


def _route_facts(obj):
    """Return (path, methods, dependant) for a route or an effective context."""
    path = getattr(obj, "path", None)
    dependant = getattr(obj, "dependant", None)
    methods = sorted(getattr(obj, "methods", None) or [])
    if path is None:
        original = getattr(obj, "original_route", None)
        if original is not None:
            path = getattr(original, "path", None)
            methods = methods or sorted(getattr(original, "methods", None) or [])
            dependant = dependant or getattr(original, "dependant", None)
    return path, methods, dependant


def iter_routes(app):
    """Yield (path, methods, dependant) for every route the app can serve."""
    from fastapi.routing import APIRoute, _IncludedRouter

    for route in app.routes:
        if isinstance(route, APIRoute):
            yield _route_facts(route)
            continue
        if not isinstance(route, _IncludedRouter):
            continue
        # FastAPI defers inclusion, so the real routes (carrying the merged
        # parent-router dependencies) only exist inside the included router.
        seen: set[tuple] = set()
        for getter in ("effective_candidates", "effective_low_priority_routes"):
            resolve = getattr(route, getter, None)
            if resolve is None:
                continue
            try:
                contexts = resolve() or []
            except Exception as exc:  # noqa: BLE001
                raise RatchetError(
                    f"could not resolve {getter} for an included router: {exc}"
                ) from exc
            for context in contexts:
                facts = _route_facts(context)
                key = (facts[0], tuple(facts[1]))
                if not facts[0] or key in seen:
                    continue
                seen.add(key)
                yield facts


def _walk(dependant, seen=None):
    if seen is None:
        seen = set()
    if dependant is None or id(dependant) in seen:
        return
    seen.add(id(dependant))
    yield dependant
    for sub in dependant.dependencies or []:
        yield from _walk(sub, seen)


def is_authenticated(dependant) -> bool:
    for node in _walk(dependant):
        call = node.call
        if call is None:
            continue
        qualname = getattr(call, "__qualname__", getattr(call, "__name__", "")) or ""
        factory = qualname.split(".<locals>.")[0].split(".")[-1]
        leaf = qualname.split(".")[-1]
        if factory in AUTHORIZER_FACTORIES:
            return True
        if leaf in AUTHENTICATORS or factory in AUTHENTICATORS:
            return True
    return False


def unauthenticated_routes(app) -> list[str]:
    found = set()
    for path, methods, dependant in iter_routes(app):
        if not path or is_authenticated(dependant):
            continue
        found.add(f"{','.join(methods) or 'ANY'} {path}")
    return sorted(found)


def read_baseline(path: Path = BASELINE_PATH) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RatchetError(f"could not read the baseline at {path}: {exc}") from exc
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current unauthenticated routes as the new baseline",
    )
    args = parser.parse_args(argv)

    app = _load_app()
    current = unauthenticated_routes(app)

    if args.write_baseline:
        BASELINE_PATH.write_text(
            "# Routes that carry no authentication dependency.\n"
            "# Public by design (login, health, HMAC webhooks) or not yet fixed.\n"
            "# This list may shrink. It may never grow.\n"
            "# Regenerate with: python Helper_Scripts/ci/route_auth_ratchet.py --write-baseline\n"
            + "\n".join(current)
            + "\n",
            encoding="utf-8",
        )
        print(f"Recorded {len(current)} unauthenticated routes.")
        return 0

    baseline = read_baseline()
    added = sorted(set(current) - baseline)
    if added:
        print("New routes with no authentication dependency:", file=sys.stderr)
        for route in added:
            print(f"  {route}", file=sys.stderr)
        print(
            "\nAdd an auth dependency, or -- if the route is public on purpose --\n"
            "regenerate the baseline and say why in the commit message.",
            file=sys.stderr,
        )
        return 1

    removed = len(baseline) - len(set(current) & baseline)
    print(f"{len(current)} unauthenticated routes ({removed} fewer than baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
