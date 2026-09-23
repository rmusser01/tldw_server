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
from typing import Any, Iterator

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

# Callables that *unconditionally* establish who the caller is.  A route reaching
# any of these has been authenticated; what it then does with the identity is a
# different gate's problem.
#
# Membership is deliberately strict.  A guard that admits anonymous callers on
# any branch does not belong here, however protective its name -- see
# NOT_AUTHENTICATION below.
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
        "require_service_principal",
        # Explicitly decoupled from loopback checks; resolves an admin principal.
        "require_shared_audio_installer_access",
        "resolve_user_id_for_request",
        "verify_api_key",
        "verify_jwt_and_fetch_user",
        "verify_prompts_user",
        # Always get_auth_principal() + SYSTEM_CONFIGURE.
        "_require_system_configure_access",
    }
)

# Guards that read like authentication and are not.  Kept as a named list so the
# distinction is reviewable, and pinned by a test.
#
#   rbac_rate_limit("sharing.admin")  -- a requests-per-minute budget.  Three
#       /sharing/admin/* routes shipped with this as their only guard.
#   require_local_setup_access        -- authorizes by network locality
#       (loopback client, local Host header).  An anonymous caller on localhost
#       passes, so setup routes using it are anonymous-capable by design and
#       belong in the baseline where a human can see them.  The
#       _require_*_setup_* guards below delegate to it before setup completes.
NOT_AUTHENTICATION = frozenset(
    {
        "check_auth_rate_limit",
        "check_evaluation_rate_limit",
        "check_rate_limit",
        "get_rate_limiter_dep",
        "kanban_rate_limit",
        "rbac_rate_limit",
        "require_local_setup_access",
        "_require_first_run_write_access",
        "_require_mcp_tools_catalog_access",
        "_require_mcp_tools_setup_or_admin_access",
        "_require_setup_write_access",
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


def _load_app() -> Any:
    """Build the FastAPI app with every route family enabled.

    Mutates ``os.environ`` (route policy, ``AUTH_MODE``, ``TEST_MODE``) and
    prepends the repository root to ``sys.path``, so it must run in a process
    that has not already imported the app.

    Returns the ``FastAPI`` instance.
    Raises ``RatchetError`` if the app cannot be constructed.
    """
    # Test-mode wiring swaps in auth shims, so a run parented by pytest sees a
    # different dependency tree than a bare one and the inventory stops being
    # reproducible. Measure the production wiring, always.
    for marker in ("MINIMAL_TEST_APP", "PYTEST_CURRENT_TEST", "TEST_MODE", "TLDW_TEST_MODE"):
        os.environ.pop(marker, None)
    for key, value in ROUTE_POLICY_ENV.items():
        os.environ[key] = value
    os.environ.setdefault("AUTH_MODE", "single_user")
    # Config validation refuses to build without one. The app is inspected, never
    # served, so this value authenticates nothing.
    os.environ.setdefault(
        "SINGLE_USER_API_KEY", "route-auth-ratchet-inspection-only-0000000000000000"
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from tldw_Server_API.app.main import app
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        raise RatchetError(f"could not build the FastAPI app: {exc}") from exc
    return app


def _route_facts(obj: Any) -> tuple[str | None, list[str], Any]:
    """Return ``(path, methods, dependant)`` for a route or an effective context.

    Falls back to ``original_route`` for contexts that do not carry the fields
    directly.  ``path`` is ``None`` when it cannot be determined.
    """
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


def iter_routes(app: Any) -> Iterator[tuple[str | None, list[str], Any]]:
    """Yield ``(path, methods, dependant)`` for every route the app can serve.

    Raises ``RatchetError`` if an included router refuses to resolve, rather
    than silently under-reporting the route inventory.
    """
    from fastapi.routing import APIRoute

    for route in app.routes:
        if isinstance(route, APIRoute):
            yield _route_facts(route)
            continue
        # Older FastAPI releases flatten included routers into APIRoute
        # instances and do not expose the private _IncludedRouter type.
        if not any(
            hasattr(route, name)
            for name in ("effective_candidates", "effective_low_priority_routes")
        ):
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


def _walk(dependant: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """Yield ``dependant`` and every sub-dependant, visiting each at most once.

    ``seen`` carries traversal state across the recursion; callers omit it.
    """
    if seen is None:
        seen = set()
    if dependant is None or id(dependant) in seen:
        return
    seen.add(id(dependant))
    yield dependant
    for sub in dependant.dependencies or []:
        yield from _walk(sub, seen)


def is_authenticated(dependant: Any) -> bool:
    """Return True when the dependency tree reaches a known authenticator.

    Matches the closure's defining factory as well as the callable itself,
    because ``require_roles("admin")`` yields a closure named ``_checker``.
    """
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


def unauthenticated_routes(app: Any) -> list[str]:
    """Return sorted ``"METHOD[,METHOD] /path"`` entries for unguarded routes."""
    found: set[str] = set()
    for path, methods, dependant in iter_routes(app):
        if not path or is_authenticated(dependant):
            continue
        found.add(f"{','.join(methods) or 'ANY'} {path}")
    return sorted(found)


def read_baseline(path: Path = BASELINE_PATH) -> set[str]:
    """Return the reviewed baseline entries, ignoring blanks and comments.

    Raises ``RatchetError`` if the file cannot be read, so a missing or
    unreadable baseline fails loudly instead of passing vacuously.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RatchetError(f"could not read the baseline at {path}: {exc}") from exc
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }


def diff_against_baseline(
    current: list[str] | set[str], baseline: set[str]
) -> tuple[list[str], list[str]]:
    """Return ``(added, stale)`` entries, each sorted.

    ``added`` are routes that lost (or never had) authentication.  ``stale`` are
    baseline entries whose routes have since gained it or been removed -- these
    fail too, because a listed exception that no longer describes the tree can
    be silently reused by a later regression on the same method and path.
    """
    live = set(current)
    return sorted(live - baseline), sorted(baseline - live)


def main(argv: list[str] | None = None) -> int:
    """Run the ratchet. Returns 0 when the baseline holds, 1 when it does not.

    Writes diagnostics to stdout and stderr rather than the application logger:
    this is a CI command whose streams are its contract, matching the sibling
    ``rls_coverage_ratchet.py``.
    """
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
            "# Routes whose dependency tree reaches no authenticator.\n"
            "#\n"
            "# Three kinds live here:\n"
            "#   1. Public by design -- login/register/reset, health, signed\n"
            "#      webhooks, token-bearer share links.\n"
            "#   2. Guarded by something that is not authentication -- the\n"
            "#      /setup/* routes authorize by network locality (loopback\n"
            "#      client, local Host header), so any anonymous caller that can\n"
            "#      reach loopback passes. Defensible for first-run setup, but it\n"
            "#      is not identity, so it is listed rather than hidden.\n"
            "#   3. Not yet fixed.\n"
            "#\n"
            "# The list must match the tree exactly: entries may be removed when a\n"
            "# route gains auth, and adding one is a reviewable diff.\n"
            "# Regenerate with: python Helper_Scripts/ci/route_auth_ratchet.py --write-baseline\n"
            + "\n".join(current)
            + "\n",
            encoding="utf-8",
        )
        print(f"Recorded {len(current)} unauthenticated routes.")
        return 0

    baseline = read_baseline()
    added, stale = diff_against_baseline(current, baseline)

    if added:
        print("New routes with no authentication dependency:", file=sys.stderr)
        for route in added:
            print(f"  {route}", file=sys.stderr)
        print(
            "\nAdd an auth dependency, or -- if the route is public on purpose --\n"
            "regenerate the baseline and say why in the commit message.",
            file=sys.stderr,
        )

    if stale:
        # A baseline entry whose route has gained auth (or been deleted) is a
        # loaded gun: leave it listed, and removing the dependency again later
        # produces no *added* entry, so CI stays green while the route goes
        # public. The list has to shrink when the tree does.
        print(
            "Baseline entries that are no longer unauthenticated:", file=sys.stderr
        )
        for route in stale:
            print(f"  {route}", file=sys.stderr)
        print(
            "\nThese routes gained an auth dependency or no longer exist. Run\n"
            "  python Helper_Scripts/ci/route_auth_ratchet.py --write-baseline\n"
            "so the exception cannot be silently reused later.",
            file=sys.stderr,
        )

    if added or stale:
        return 1

    print(f"{len(current)} unauthenticated routes; baseline exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
