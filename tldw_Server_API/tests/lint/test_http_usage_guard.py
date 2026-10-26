"""Guard: outbound HTTP goes through core.http_client (TASK-13950.11).

``tldw_Server_API.app.core.http_client`` holds the egress policy, DNS pinning,
``trust_env=False`` and the TLS floor. A module that builds its own HTTP client (httpx,
requests, aiohttp, urllib.request, http.client) skips all of them. This guard fails on
any such use in tldw_Server_API/app (``tests`` directories excluded) outside ALLOWLIST.
It always runs: there is no env gate.

What counts as a use (the only definition; ``uses`` implements it): any value read of
``<library>[.<submodule>...].<name>`` for a library and name in TARGETS (the clients,
sessions and request functions), resolved through the file's imports:
``import httpx``, ``import httpx as hx``, ``from httpx import AsyncClient as C``,
``import requests.sessions``, ``from urllib import request``, ``from httpx import *``,
and imports inside functions. A call is a use, and so is storing or passing the class
(``factory = httpx.AsyncClient``). Type annotations, exception classes
(``httpx.HTTPError``, ``aiohttp.ClientError``) and other members (``httpx.Timeout``) are
not. Imports are collected file-wide, so a name imported in one function counts in every
function of the file (it fails closed).
Known limits: ``importlib.import_module("httpx")`` is not seen. urllib3 is not listed:
app code does not use it directly.

ALLOWLIST maps a repo-relative path to why it may use an HTTP client library directly:
why core.http_client cannot apply, or ``"migrate: ..."`` for an offender a later PR
moves onto it. The test fails on a use in an unlisted file, on an entry whose file is
gone or no longer uses one directly (delete it), and on an entry with no reason.
Adding an entry is a bump-with-reason, reviewed in the PR.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT, parse_tree

pytestmark = pytest.mark.unit

GUARD = "tldw_Server_API/tests/lint/test_http_usage_guard.py"
_VERBS = frozenset({"request", "get", "post", "put", "patch", "delete", "head", "options"})
# Library module -> its clients, sessions and request functions (the entry points).
TARGETS: dict[str, frozenset[str]] = {
    "httpx": _VERBS | {"Client", "AsyncClient", "stream"},
    "requests": _VERBS | {"Session", "session"},
    "aiohttp": frozenset({"ClientSession", "request"}),
    "urllib.request": frozenset({"urlopen", "Request", "build_opener", "urlretrieve"}),
    "http.client": frozenset({"HTTPConnection", "HTTPSConnection"}),
}

_ASGI = (
    "In-process dispatch: an httpx.ASGITransport client posts to this app's own "
    "/api/v1/chat/completions. No socket is opened, so there is no egress to police."
)
ALLOWLIST: dict[str, str] = {
    # Permanent: core.http_client cannot apply.
    "tldw_Server_API/app/core/http_client.py": (
        "The egress-policy client itself: create_client/create_async_client build the httpx "
        "clients every other module must use."
    ),
    "tldw_Server_API/app/core/Buddy/turns.py": _ASGI,
    "tldw_Server_API/app/core/Persona/live_conversation.py": _ASGI,
    "tldw_Server_API/app/core/MCP_unified/browser_cdp/client.py": (
        "CDP transport: GETs /json from an operator-configured Chrome debugger, loopback-only "
        "unless allow_non_loopback is set. The CDP session itself runs over a websockets "
        "connection core.http_client cannot carry, so moving the GET alone gains nothing."
    ),
    "tldw_Server_API/app/core/TTS/adapters/vibevoice_realtime_adapter.py": (
        "WebSocket transport: aiohttp ClientSession.ws_connect to the realtime TTS backend, which "
        "core.http_client cannot carry. It already runs evaluate_url_policy on the URL and pins the "
        "resolved IPs (_PinnedWebSocketResolver)."
    ),
    "tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py": (
        "Loopback-only local sidecar this process spawns and supervises (validate_loopback_host, "
        "trust_env=False, per-run token). The egress policy is for outbound URLs and refuses "
        "its 127.0.0.1 port by default."
    ),
    # To migrate onto core.http_client, one PR each (TASK-13950.11 follow-up slices).
    "tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_transports/sse.py": (
        "migrate: MCP-over-SSE client for a user-configured remote URL. Raw httpx.AsyncClient "
        "with only its own SSRF check (validate_mcp_http_url); use create_async_client/astream_sse."
    ),
    "tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_transports/streamable_http.py": (
        "migrate: MCP streamable-HTTP client for a user-configured remote URL. Raw httpx.AsyncClient "
        "with only its own SSRF check (validate_mcp_http_url); use create_async_client/afetch."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py": (
        "migrate: the unplanned vLLM branch posts with a raw httpx.Client; the planned branch beside "
        "it already uses create_client + fetch(configured_endpoint=...)."
    ),
    "tldw_Server_API/app/core/MCP_unified/modules/implementations/web_fetch_module.py": (
        "migrate: fetches user-supplied URLs. It re-checks the web outbound policy on every redirect "
        "hop but connects through a raw httpx.AsyncClient that honours env proxies; use astream_bytes."
    ),
    "tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py": (
        "migrate: audiocpp_server client. Loopback by default, but allow_remote_base_url sends to any "
        "host through a raw httpx.AsyncClient with no egress policy; use create_async_client/afetch."
    ),
    "tldw_Server_API/app/core/Workflows/adapters/integration/github.py": (
        "migrate: creates GitHub issues via POST https://api.github.com with a raw httpx.AsyncClient "
        "that honours env proxies; use apost/afetch."
    ),
    "tldw_Server_API/app/services/admin_webhooks_service.py": (
        "migrate: delivers admin webhooks to registered URLs through a raw httpx.AsyncClient "
        "(_default_async_client_factory) with no egress check per attempt; use afetch."
    ),
    "tldw_Server_API/app/core/Audio_Studio/providers/ace_step.py": (
        "migrate: defaults client_factory to httpx.AsyncClient (a reference the regex guard missed). "
        "It checks its own origin allowlist, not the egress policy; default to create_async_client."
    ),
    "tldw_Server_API/app/core/Slides/standalone_html_provider.py": (
        "migrate: streams provider completions through _AsyncClient = httpx.AsyncClient (an alias the "
        "regex guard missed), with no egress policy; use create_async_client with a configured endpoint."
    ),
    "tldw_Server_API/app/core/Workflows/adapters/integration/podcast_rss.py": (
        "migrate: fetches a workflow-supplied source_feed_url with urllib urlopen: scheme check only, "
        "no egress policy, redirects followed, body unbounded. Use fetch with a byte cap."
    ),
    "tldw_Server_API/app/core/Ingestion_Sources/git_repository.py": (
        "migrate: GitHub API sync via urllib urlopen. Only the first URL is pinned to api.github.com; "
        "urlopen follows redirects and honours env proxies. Use fetch/fetch_json."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py": (
        "migrate: http.client readiness probe (GET /health) for the managed OCR server. The host "
        "defaults to 127.0.0.1 but comes unchecked from CHATLLM_OCR_HOST, LLAMACPP_OCR_HOST or the "
        "OCR profile; use create_client/fetch."
    ),
}


# --- The scan -----------------------------------------------------------------------------------


def _imports(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Every import statement in ``tree``, at any depth. Only statements can hold one."""
    stack: list[ast.AST] = [tree]
    while stack:
        for child in ast.iter_child_nodes(stack.pop()):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child
            elif isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                stack.append(child)


def _library(module: str) -> str | None:
    """The TARGETS library ``module`` is, or is inside."""
    return next((lib for lib in TARGETS if module == lib or module.startswith(f"{lib}.")), None)


def _related(dotted: str) -> bool:
    """``dotted`` is a library, inside one, or a package above one (``urllib``)."""
    return _library(dotted) is not None or any(lib.startswith(f"{dotted}.") for lib in TARGETS)


def _bindings(tree: ast.Module) -> dict[str, str]:
    """Local name -> the dotted library object an import binds it to."""
    bound: dict[str, str] = {}
    for node in _imports(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c`` to ``a.b``.
                target = alias.name if alias.asname else alias.name.partition(".")[0]
                if _related(target):
                    bound[alias.asname or target] = target
        elif node.level == 0 and node.module:
            star = TARGETS.get(_library(node.module) or "", frozenset())
            for alias in node.names:
                for name in sorted(star) if alias.name == "*" else [alias.name]:
                    if _related(f"{node.module}.{name}"):
                        bound[alias.asname or name] = f"{node.module}.{name}"
    return bound


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id, *reversed(parts)] if isinstance(node, ast.Name) else None


def uses(tree: ast.Module) -> list[str]:
    """``"line N: httpx.AsyncClient"`` for every use in ``tree``; see the module docstring."""
    bound = _bindings(tree)
    if not bound:
        return []
    annotations: set[int] = set()
    for node in ast.walk(tree):
        for annotation in (getattr(node, "annotation", None), getattr(node, "returns", None)):
            if isinstance(annotation, ast.AST):
                annotations.update(map(id, ast.walk(annotation)))
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        if id(node) in annotations:
            continue
        path = _dotted(node)
        if path and path[0] in bound:
            *module, name = [*bound[path[0]].split("."), *path[1:]]
            if name in TARGETS.get(_library(".".join(module)) or "", ()):
                found.add((node.lineno, ".".join([*module, name])))
    return [f"line {line}: {name}" for line, name in sorted(found)]


def scan(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Repo-relative path -> its uses, for every file under tldw_Server_API/app that has one."""
    repo_root = repo_root.resolve()
    app = repo_root / "tldw_Server_API" / "app"
    found: dict[str, list[str]] = {}
    for path, tree in parse_tree(app).items():
        if "tests" not in path.relative_to(app).parts and (hits := uses(tree)):
            found[path.relative_to(repo_root).as_posix()] = hits
    return found


def problems(found: dict[str, list[str]], allowlist: dict[str, str]) -> list[str]:
    """Unlisted uses, stale entries and entries without a reason, each with its fix."""
    out = [
        f"{path} uses an HTTP client library directly ({'; '.join(found[path])}). Send outbound "
        "HTTP through tldw_Server_API.app.core.http_client (create_async_client/create_client, "
        "afetch/fetch, astream_sse, astream_bytes) so the egress policy applies. If it cannot (an in-process "
        f"ASGI call, a loopback-only sidecar), add an entry to ALLOWLIST in {GUARD} with the "
        f'reason; it is reviewed in the PR:\n    "{path}": "<why core.http_client cannot apply>",'
        for path in sorted(found.keys() - allowlist.keys())
    ]
    out += [
        f"ALLOWLIST entry {path} is stale: the file is gone or no longer uses an HTTP "
        f"client library directly. Delete it from {GUARD}."
        for path in sorted(allowlist.keys() - found.keys())
    ]
    out += [
        f"ALLOWLIST entry {path} has no reason. Say why core.http_client cannot apply, or start "
        "it with 'migrate: ' for an offender a later PR moves onto it."
        for path, reason in sorted(allowlist.items())
        if not reason.strip()
    ]
    return out


def test_outbound_http_goes_through_core_http_client() -> None:
    found = problems(scan(), ALLOWLIST)
    assert not found, "\n\n".join(found)


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / "tldw_Server_API" / "app" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def _names(found: dict[str, list[str]], path: str) -> list[str]:
    return [hit.split(": ", 1)[1] for hit in found.get(path, [])]


def test_a_seeded_aliased_from_import_fails(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {
            "core/sneaky.py": """\
                from httpx import AsyncClient as C


                async def go() -> None:
                    async with C() as client:
                        await client.get("https://example.com")
                """
        },
    )
    (problem,) = problems(scan(root), {})
    assert problem.startswith(
        "tldw_Server_API/app/core/sneaky.py uses an HTTP client library directly (line 5: httpx.AsyncClient)"
    ), problem
    assert "tldw_Server_API.app.core.http_client" in problem
    assert GUARD in problem and '"tldw_Server_API/app/core/sneaky.py": "<why' in problem


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import httpx\nhttpx.AsyncClient()\n", "httpx.AsyncClient"),
        ("import httpx as hx\nhx.Client()\n", "httpx.Client"),
        ("from requests import Session\nSession()\n", "requests.Session"),
        ("import requests.sessions\nrequests.sessions.Session()\n", "requests.sessions.Session"),
        ("from requests import sessions as s\ns.session()\n", "requests.sessions.session"),
        ("def f():\n    import requests\n    return requests.post('u')\n", "requests.post"),
        ("from httpx import *\nstream('GET', 'u')\n", "httpx.stream"),
        # A reference is a use too: ace_step.py and standalone_html_provider.py build clients this way.
        ("import httpx\nclient_factory = httpx.AsyncClient\n", "httpx.AsyncClient"),
        # aiohttp and urllib.request bypass the same egress policy.
        ("from aiohttp import ClientSession as S\nS()\n", "aiohttp.ClientSession"),
        ("def f():\n    import aiohttp\n    return aiohttp.request('GET', 'u')\n", "aiohttp.request"),
        ("import urllib.request\nurllib.request.urlopen('u')\n", "urllib.request.urlopen"),
        ("from urllib import request as r\nr.build_opener()\n", "urllib.request.build_opener"),
        ("from urllib.request import Request as R\nR('u')\n", "urllib.request.Request"),
        ("import http.client\nhttp.client.HTTPSConnection('h')\n", "http.client.HTTPSConnection"),
    ],
    ids=[
        *("import", "import-as", "from-import", "submodule", "submodule-as", "in-function", "star", "reference"),
        *("aiohttp-from-as", "aiohttp-in-function", "urllib-import", "urllib-from-submodule", "urllib-from-as"),
        "http-client",
    ],
)
def test_every_import_form_is_seen(tmp_path: Path, source: str, expected: str) -> None:
    assert _names(scan(_seed(tmp_path, {"core/m.py": source})), "tldw_Server_API/app/core/m.py") == [expected]


def test_annotations_exceptions_other_members_and_lookalikes_are_not_uses(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {
            "core/typed.py": """\
                import httpx


                def f(client: httpx.AsyncClient) -> httpx.Client:
                    kept: httpx.AsyncClient = client
                    try:
                        return httpx.Timeout(5)
                    except httpx.HTTPError:
                        raise
                """,
            # Resource_Governance/policy_eval.py's shape: a dict named ``requests``. The regex flagged it.
            "core/lookalike.py": "requests = {'burst': 2.0}\nburst = requests.get('burst')\n",
            # Exception classes and helpers only (paper_search.py, eval_runner.py): not clients.
            "core/errors_only.py": """\
                import aiohttp
                import urllib.parse
                import urllib.request
                from urllib.error import HTTPError

                ERRORS = (aiohttp.ClientError, HTTPError)
                URL = urllib.parse.urljoin("https://a/", urllib.request.pathname2url("b"))
                """,
            "core/tests/test_x.py": "import httpx\nhttpx.Client()\n",
        },
    )
    assert scan(root) == {}


def test_stale_and_reasonless_entries_fail(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {"core/a.py": "import httpx\nhttpx.get('u')\n", "core/clean.py": "import httpx\nhttpx.Timeout(1)\n"},
    )
    found = scan(root)
    out = problems(
        found,
        {
            "tldw_Server_API/app/core/a.py": " ",
            "tldw_Server_API/app/core/clean.py": "migrate: already moved",
            "tldw_Server_API/app/core/gone.py": "migrate: file deleted",
        },
    )

    assert [line.split(" ", 3)[2] for line in out] == [
        "tldw_Server_API/app/core/clean.py",
        "tldw_Server_API/app/core/gone.py",
        "tldw_Server_API/app/core/a.py",
    ], out
    assert "is stale" in out[0] and "is stale" in out[1] and "has no reason" in out[2]
    assert problems(found, {"tldw_Server_API/app/core/a.py": "loopback-only sidecar"}) == []
