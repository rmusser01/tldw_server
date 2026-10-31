"""Guard: outbound HTTP goes through core.http_client (TASK-13950.11).

``tldw_Server_API.app.core.http_client`` holds the egress policy, DNS pinning,
``trust_env=False`` and the TLS floor. A module that builds its own HTTP client (httpx,
requests, aiohttp, urllib.request, http.client, curl_cffi) skips all of them. This guard fails on
any such use in tldw_Server_API/app (``tests`` directories excluded) outside ALLOWLIST.
It always runs: there is no env gate.

What counts as a use (the only definition; ``uses`` implements it): any value read of
``<library>[.<submodule>...].<name>`` for a library and name in TARGETS (the clients,
sessions and request functions), resolved through the file's imports:
``import httpx``, ``import httpx as hx``, ``from httpx import AsyncClient as C``,
``import requests.sessions``, ``from urllib import request``, ``from httpx import *``,
and imports inside functions. So is a library reached through another imported name's
attributes (``from ...core import http_client as hc; hc.aiohttp.ClientSession()``).
A call is a use, and so is storing or passing the class (``factory = httpx.AsyncClient``).
These are not: type annotations, the class argument of ``isinstance``/``issubclass``, the
type argument of ``cast``, exception classes (``httpx.HTTPError``, ``aiohttp.ClientError``)
and other members (``httpx.Timeout``). Imports are collected file-wide, so a name imported
in one function counts in every function of the file (it fails closed).
Known limits: ``importlib.import_module("httpx")``, ``getattr(httpx, "AsyncClient")(...)``,
module rebinding (``m = httpx; m.AsyncClient()``) and a library re-exported under its own
name (``from pkg import httpx``) are not seen. A factory in an allowlisted file hides its
callers: ``omnivoice_adapter.py`` gets a raw client from ``create_sidecar_async_client``.
urllib3 is not listed: app code does not use it directly.

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
    "curl_cffi.requests": _VERBS | {"Session", "AsyncSession"},
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
        "its 127.0.0.1 port by default. Its public create_sidecar_async_client() factory also "
        "hands omnivoice_adapter.py a raw httpx.AsyncClient for the same loopback sidecar."
    ),
    "tldw_Server_API/app/core/Web_Scraping/preflight/adapters/http.py": (
        "Impersonating probe transport: curl_cffi AsyncSession, which core.http_client has no async "
        "path for. Each dispatch takes a fresh ProbeEgressGuard decision, pins DNS to the decided "
        "IPs (CURLOPT_RESOLVE), refuses proxies and re-checks every redirect hop."
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


def _is_target(parts: list[str]) -> bool:
    """``parts`` is a library module path (or one inside it) followed by one of its entry points."""
    return parts[-1] in TARGETS.get(_library(".".join(parts[:-1])) or "", ())


def _bindings(tree: ast.Module) -> dict[str, str]:
    """Local name -> the dotted object an import binds it to, for every import in the file."""
    bound: dict[str, str] = {}
    for node in _imports(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c`` to ``a.b``.
                target = alias.name if alias.asname else alias.name.partition(".")[0]
                bound[alias.asname or target] = target
        else:
            # A relative import keeps its leading dots, so it never resolves to a library.
            module = "." * node.level + (f"{node.module}." if node.module else "")
            star = TARGETS.get(_library(node.module or "") or "", frozenset()) if not node.level else frozenset()
            for alias in node.names:
                for name in sorted(star) if alias.name == "*" else [alias.name]:
                    bound[alias.asname or name] = module + name
    return bound


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id, *reversed(parts)] if isinstance(node, ast.Name) else None


# Calls whose argument at this index is only read as a type, like an annotation.
_TYPE_ARGUMENT = {"isinstance": 1, "issubclass": 1, "cast": 0}


def _type_only(tree: ast.Module) -> set[int]:
    """ids of nodes only read as types: annotations, isinstance/issubclass classes, cast's type."""
    roots: list[object] = []
    for node in ast.walk(tree):
        roots += [getattr(node, "annotation", None), getattr(node, "returns", None)]
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            index = _TYPE_ARGUMENT.get(name)
            if index is not None and len(node.args) > index:
                roots.append(node.args[index])
    return {id(child) for root in roots if isinstance(root, ast.AST) for child in ast.walk(root)}


def uses(tree: ast.Module) -> list[str]:
    """``"line N: httpx.AsyncClient"`` for every use in ``tree``; see the module docstring."""
    bound = _bindings(tree)
    if not bound:
        return []
    candidates: list[tuple[ast.AST, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        path = _dotted(node)
        if not path or path[0] not in bound:
            continue
        resolved = [*bound[path[0]].split("."), *path[1:]]
        if _is_target(resolved):  # ``httpx.AsyncClient``, ``C`` from ``from httpx import AsyncClient as C``
            candidates.append((node, node.lineno, ".".join(resolved)))
        elif any(_is_target(path[i:]) for i in range(1, len(path) - 1)):  # ``hc.httpx.AsyncClient``
            candidates.append((node, node.lineno, ".".join(path)))
    if not candidates:
        return []
    type_only = _type_only(tree)
    found = {(line, name) for node, line, name in candidates if id(node) not in type_only}
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


# --- Pin: who may pass configured_endpoint= -------------------------------------------------------
# ``configured_endpoint=<scope>`` is the private-network opt-in: it lets one exact origin through the
# egress policy. Only a module whose origin comes from operator configuration or a code-owned catalog
# may build a scope, never from a user-supplied or server-discovered URL. The scan fails on any call
# keyword ``configured_endpoint=<expr>``, and on a ``"configured_endpoint"`` dict-literal key or
# subscript store, outside SCOPE_ALLOWLIST. Only a *parameter* of an enclosing function passed on under
# its own name (``configured_endpoint=configured_endpoint``) is plumbing and exempt; a local variable of
# that name is a source, and so is any rebind of the parameter to something other than ``None``. A dict
# built elsewhere and ``**``-splatted is seen only where its literal key, key store or
# ``setdefault("configured_endpoint", ...)`` is. Known limits: dict comprehensions and ``dict([...])``
# pairs that spell the key are not seen. Other scope carriers (``required_endpoint_scope=``, ``TrustedProviderEndpoint(scope=...)``,
# ``provider_credential_runtime``) are gated through the allowlisted adapters they feed.
SCOPE_ALLOWLIST: dict[str, str] = {
    "tldw_Server_API/app/api/v1/endpoints/llm_providers.py": (
        "scope built from the configured provider endpoint URL for the provider calls it makes."
    ),
    "tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py": (
        "scope required by the strict call policy (required_endpoint_scope), checked against the request URL."
    ),
    "tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py": (
        "scope from the operator-configured custom endpoint (endpoint.scope)."
    ),
    "tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py": (
        "scope from the operator-configured local-LLM (llama.cpp, Ollama, ...) base URL."
    ),
    "tldw_Server_API/app/core/Local_LLM/llamacpp_snapshot_operations.py": (
        "scope from the configured llama.cpp server origin."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py": (
        "scope from the operator-configured Qwen3-ASR server URL."
    ),
    "tldw_Server_API/app/api/v1/endpoints/setup.py": (
        "POST /first-run/providers/validate scopes the request payload's base_url so a local provider can be "
        "probed. require_local_setup_access admits only a loopback client with a local Host header and any "
        "proxy client-IP headers resolving to loopback, or an admin when remote setup is explicitly enabled. "
        "Reachable only while needs_setup is true and the first-run state is non-terminal (409 afterwards). "
        "No ordinary authenticated user can reach it."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/llamacpp_ocr.py": (
        "scope from the operator-configured LLAMACPP_OCR_HOST/PORT (remote) or the managed local runtime's "
        "own host/port."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_AudioCpp.py": (
        "scope from the operator-configured audio.cpp server origin, passed to afetch through request_kwargs."
    ),
    "tldw_Server_API/app/core/Slides/standalone_html_provider.py": (
        "scope from the code-owned loopback adapter catalog (llama.cpp / Ollama), and for the official OpenAI and "
        "Anthropic HTTPS origins the discovery call policy (required_endpoint_scope, built from the catalog's "
        "endpoint_identity); never a request URL. Generation keeps the public-only policy for those origins."
    ),
    "tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_transports/sse.py": (
        "scope from the operator-configured MCP SSE origin (sse_url, allow_private_network) via "
        "mcp_http_egress_scope; a server-supplied POST URL on another origin gets no scope."
    ),
    "tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_transports/streamable_http.py": (
        "scope from the operator-configured MCP streamable-HTTP origin (allow_private_network) via "
        "mcp_http_egress_scope."
    ),
    "tldw_Server_API/app/core/Audio_Studio/providers/ace_step.py": (
        "scope from the operator's AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST origin; a legacy redirect hop "
        "is scoped only after it matches an allowlisted origin exactly."
    ),
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py": (
        "scope from the operator-configured OCR host/port, OCR profile or CHATLLM_OCR_HEALTHCHECK_URL."
    ),
    "tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py": (
        "scope from operator config (a loopback base_url, or providers.audio_cpp.extra_params."
        "allow_private_network), or an admin BYOK base_url when audio_cpp is in "
        "BYOK_ALLOWED_BASE_URL_PROVIDERS."
    ),
    "tldw_Server_API/app/core/LLM_Calls/gateway.py": (
        "model-discovery scope from provider_model_inventory.inventory_endpoint_scope: only the provider base "
        "the server's own config and environment resolve to; caller app config, BYOK and per-request bases "
        "get no scope."
    ),
    "tldw_Server_API/app/core/LLM_Calls/provider_model_inventory.py": (
        "discovery scope is the caller's inventory_endpoint_scope (server-config base only) or the strict call "
        "policy's required_endpoint_scope, the same decision generation makes for that endpoint."
    ),
}


_SCOPE_KEY = "configured_endpoint"


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    args = node.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    return names


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class _ScopeVisitor(ast.NodeVisitor):
    """Collect lines that set a configured_endpoint scope; a parameter passed on by name is plumbing."""

    def __init__(self) -> None:
        self.params: list[set[str]] = []
        self.lines: list[int] = []
        self.plumbing: set[int] = set()  # id() of subscript targets assigned a parameter
        self.narrowing: set[int] = set()  # id() of Name targets assigned the literal None

    def _is_parameter(self, value: ast.AST) -> bool:
        return isinstance(value, ast.Name) and value.id == _SCOPE_KEY and any(_SCOPE_KEY in p for p in self.params)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        self.params.append(_parameter_names(node))  # closures see every enclosing function's parameters
        self.generic_visit(node)
        self.params.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:
        self.lines += [node.lineno for kw in node.keywords if kw.arg == _SCOPE_KEY and not self._is_parameter(kw.value)]
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == _SCOPE_KEY
            and not (len(node.args) > 1 and self._is_parameter(node.args[1]))
        ):
            self.lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Any rebind of the name, even of a parameter, is a possible new source; `= None` only narrows.
        if node.id == _SCOPE_KEY and isinstance(node.ctx, ast.Store) and id(node) not in self.narrowing:
            self.lines.append(node.lineno)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name == _SCOPE_KEY:
            self.lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == _SCOPE_KEY and not self._is_parameter(value):
                self.lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_parameter(node.value):
            self.plumbing |= {id(t) for t in node.targets if isinstance(t, ast.Subscript)}
        if _is_none(node.value):
            self.narrowing |= {id(t) for t in node.targets}
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _is_none(node.value):
            self.narrowing.add(id(node.target))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            id(node) not in self.plumbing
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == _SCOPE_KEY
        ):
            self.lines.append(node.lineno)
        self.generic_visit(node)


def scope_uses(tree: ast.Module) -> list[str]:
    """Lines that set a scope: ``configured_endpoint=<expr>``, a ``"configured_endpoint"`` dict key or store.

    Exempt only a parameter of an enclosing function passed on under its own name and never rebound to
    anything but None. A local or module variable of that name, or a rebind of the parameter (assignment,
    walrus, for/with/except target, comprehension), is a scope source and counts.
    """
    visitor = _ScopeVisitor()
    visitor.visit(tree)
    return [f"line {line}" for line in sorted(set(visitor.lines))]


def scope_scan(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Repo-relative path -> lines that set a configured_endpoint scope, under tldw_Server_API/app."""
    repo_root = repo_root.resolve()
    app = repo_root / "tldw_Server_API" / "app"
    found: dict[str, list[str]] = {}
    for path, tree in parse_tree(app).items():
        if "tests" not in path.relative_to(app).parts and (hits := scope_uses(tree)):
            found[path.relative_to(repo_root).as_posix()] = hits
    return found


def scope_problems(found: dict[str, list[str]], allowlist: dict[str, str]) -> list[str]:
    out = [
        f"{path} passes configured_endpoint= ({'; '.join(found[path])}), the private-network opt-in. "
        "Build the scope only from operator configuration or a code-owned catalog, never from a user or "
        f"server-supplied URL; then add an entry to SCOPE_ALLOWLIST in {GUARD} saying where the origin "
        "comes from:\n"
        f'    "{path}": "<where the configured origin comes from>",'
        for path in sorted(found.keys() - allowlist.keys())
    ]
    out += [
        f"SCOPE_ALLOWLIST entry {path} is stale: the file is gone or no longer sets configured_endpoint=. "
        f"Delete it from {GUARD}."
        for path in sorted(allowlist.keys() - found.keys())
    ]
    out += [
        f"SCOPE_ALLOWLIST entry {path} has no reason. Say where the configured origin comes from."
        for path, reason in sorted(allowlist.items())
        if not reason.strip()
    ]
    return out


def test_only_allowlisted_modules_set_a_configured_endpoint_scope() -> None:
    found = scope_problems(scope_scan(), SCOPE_ALLOWLIST)
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
        # A reference is a use too: a stored class (a client_factory default, an alias) builds clients later.
        ("import httpx\nclient_factory = httpx.AsyncClient\n", "httpx.AsyncClient"),
        # aiohttp and urllib.request bypass the same egress policy.
        ("from aiohttp import ClientSession as S\nS()\n", "aiohttp.ClientSession"),
        ("def f():\n    import aiohttp\n    return aiohttp.request('GET', 'u')\n", "aiohttp.request"),
        ("import urllib.request\nurllib.request.urlopen('u')\n", "urllib.request.urlopen"),
        ("from urllib import request as r\nr.build_opener()\n", "urllib.request.build_opener"),
        ("from urllib.request import Request as R\nR('u')\n", "urllib.request.Request"),
        ("import http.client\nhttp.client.HTTPSConnection('h')\n", "http.client.HTTPSConnection"),
        ("from curl_cffi.requests import AsyncSession as S\nS()\n", "curl_cffi.requests.AsyncSession"),
        # Reached through another module's attribute (the shape Audio_Transcription_AudioCpp.py used).
        ("from tldw_Server_API.app.core import http_client as hc\nhc.httpx.AsyncClient()\n", "hc.httpx.AsyncClient"),
        (
            "from tldw_Server_API.app.core import http_client as hc\nhc.aiohttp.ClientSession()\n",
            "hc.aiohttp.ClientSession",
        ),
        # cast's type argument is exempt; the client it wraps is not (one hit, on the second line).
        (
            "import httpx\nfrom typing import cast\nc = cast(\n    httpx.AsyncClient,\n    httpx.AsyncClient(),\n)\n",
            "httpx.AsyncClient",
        ),
    ],
    ids=[
        *("import", "import-as", "from-import", "submodule", "submodule-as", "in-function", "star", "reference"),
        *("aiohttp-from-as", "aiohttp-in-function", "urllib-import", "urllib-from-submodule", "urllib-from-as"),
        "http-client",
        *("curl-cffi", "reach-through-httpx", "reach-through-aiohttp", "cast-wraps-a-client"),
    ],
)
def test_every_import_form_is_seen(tmp_path: Path, source: str, expected: str) -> None:
    assert _names(scan(_seed(tmp_path, {"core/m.py": source})), "tldw_Server_API/app/core/m.py") == [expected]


def test_a_file_mixing_exception_imports_and_one_call_flags_only_the_call(tmp_path: Path) -> None:
    source = """\
        import httpx
        from httpx import HTTPError as _HTTP_ERROR

        ERRORS = (httpx.RequestError, _HTTP_ERROR)


        def fetch(url: str) -> bytes:
            try:
                return httpx.get(url).content
            except httpx.HTTPStatusError:
                raise
        """
    found = scan(_seed(tmp_path, {"core/mixed.py": source}))
    assert found == {"tldw_Server_API/app/core/mixed.py": ["line 9: httpx.get"]}


def test_annotations_exceptions_other_members_and_lookalikes_are_not_uses(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {
            "core/typed.py": """\
                import typing

                import httpx


                def f(client: httpx.AsyncClient) -> httpx.Client:
                    kept: httpx.AsyncClient = client
                    assert isinstance(kept, httpx.AsyncClient)
                    assert issubclass(type(kept), (httpx.Client, httpx.AsyncClient))
                    kept = typing.cast(httpx.AsyncClient, kept)
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
            # Reach-through to a library member that is not an entry point (AudioCpp's guard and timeout).
            "core/reach_other.py": """\
                from tldw_Server_API.app.core import http_client as hc

                if hc.aiohttp is not None:
                    TIMEOUT = hc.aiohttp.ClientTimeout(total=5)
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


# --- Pin: who may pass configured_endpoint= (the private-network opt-in) --------------------------


def test_a_seeded_module_passing_configured_endpoint_fails(tmp_path: Path) -> None:
    root = _seed(
        tmp_path,
        {
            "core/new.py": "from x import afetch, Scope\nafetch(url=u, configured_endpoint=Scope.from_url(u))\n",
            "core/passthrough.py": "def f(configured_endpoint=None):\n    g(configured_endpoint=configured_endpoint)\n",
            "core/other_kw.py": "g(configured=1, endpoint=2)\n",
        },
    )
    found = scope_scan(root)

    assert list(found) == ["tldw_Server_API/app/core/new.py"]
    out = scope_problems(found, {})
    assert len(out) == 1 and "core/new.py" in out[0] and "SCOPE_ALLOWLIST" in out[0]


def test_stale_and_reasonless_scope_entries_fail(tmp_path: Path) -> None:
    root = _seed(tmp_path, {"core/a.py": "g(configured_endpoint=s)\n", "core/clean.py": "g(x=1)\n"})
    found = scope_scan(root)
    out = scope_problems(
        found,
        {
            "tldw_Server_API/app/core/a.py": " ",
            "tldw_Server_API/app/core/clean.py": "operator endpoint",
            "tldw_Server_API/app/core/gone.py": "operator endpoint",
        },
    )

    assert "is stale" in out[0] and "clean.py" in out[0]
    assert "is stale" in out[1] and "gone.py" in out[1]
    assert "has no reason" in out[2] and "a.py" in out[2]
    assert scope_problems(found, {"tldw_Server_API/app/core/a.py": "operator endpoint"}) == []


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    configured_endpoint = Scope.from_url(u)\n    g(configured_endpoint=configured_endpoint)\n",
        "configured_endpoint = Scope.from_url(u)\ng(configured_endpoint=configured_endpoint)\n",
        "def f(other):\n    g(**{'configured_endpoint': Scope.from_url(other)})\n",
        "def f(other):\n    kw = dict(configured_endpoint=Scope.from_url(other))\n    g(**kw)\n",
        "def f(other):\n    kw = {}\n    kw['configured_endpoint'] = Scope.from_url(other)\n    g(**kw)\n",
        "def f(other):\n    kw = {}\n    kw.setdefault('configured_endpoint', Scope.from_url(other))\n    g(**kw)\n",
        "def f(configured_endpoint=None):\n    configured_endpoint = Scope.from_url(u)\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    if configured_endpoint is None:\n        configured_endpoint = Scope.from_url(u)\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    if (configured_endpoint := Scope.from_url(u)):\n        g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    for configured_endpoint in scopes:\n        g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    with open_scope() as configured_endpoint:\n        g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    configured_endpoint: object = Scope.from_url(u)\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    configured_endpoint += 1\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    def inner(u):\n        configured_endpoint = Scope.from_url(u)\n        g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    scopes = [configured_endpoint for configured_endpoint in many]\n",
    ],
    ids=[
        "local-variable",
        "module-variable",
        "dict-literal-key",
        "dict-call",
        "subscript-store",
        "setdefault",
        "shadowed-parameter",
        "fallback-rebind",
        "walrus",
        "for-target",
        "with-target",
        "annassign",
        "augassign",
        "closure-local-shadow",
        "comprehension-target",
    ],
)
def test_scope_carriers_other_than_a_parameter_pass_through_fail(tmp_path: Path, source: str) -> None:
    root = _seed(tmp_path, {"core/carrier.py": source})

    assert list(scope_scan(root)) == ["tldw_Server_API/app/core/carrier.py"]


@pytest.mark.parametrize(
    "source",
    [
        "def f(configured_endpoint=None):\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(*, configured_endpoint):\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint, /):\n    g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None):\n    def inner():\n        g(configured_endpoint=configured_endpoint)\n",
        "async def f(configured_endpoint=None):\n    kw = {'configured_endpoint': configured_endpoint}\n    await g(**kw)\n",
        "def f(configured_endpoint=None):\n    kw = {}\n    kw['configured_endpoint'] = configured_endpoint\n    g(**kw)\n",
        "class C:\n    def m(self, configured_endpoint=None):\n        return lambda: g(configured_endpoint=configured_endpoint)\n",
        "def f(configured_endpoint=None, strict=False):\n    if strict:\n        configured_endpoint = None\n    g(configured_endpoint=configured_endpoint)\n",
    ],
    ids=["default", "kwonly", "posonly", "closure", "dict-of-param", "store-of-param", "lambda-closure", "narrow-to-none"],
)
def test_a_real_parameter_pass_through_is_not_a_scope_source(tmp_path: Path, source: str) -> None:
    assert scope_scan(_seed(tmp_path, {"core/plumbing.py": source})) == {}
