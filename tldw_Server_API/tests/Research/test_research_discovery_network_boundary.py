"""End-to-end registry, attribution, and egress-boundary tests for Discovery V2."""

from __future__ import annotations

import ast
import asyncio
import builtins
import hashlib
import http.client
import importlib
import json
import socket

# Imported only so the runtime test can patch process entry points.
import subprocess  # nosec B404
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Research.discovery.contracts import (
    BudgetCeilings,
    CredentialRequirement,
    ExecutionMode,
    PredicateOperator,
    ReadinessState,
    RouteKind,
    SkippedCode,
    SourceConstraint,
    SourcePredicate,
    SourceRouteReference,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    AttemptJournal,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    LogicalOutcomeState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.planner import (
    DateIntervalQuery,
    GeneralFreeTextQuery,
    IdentifierLookupQuery,
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3]
_DISCOVERY_ROOT = Path(__file__).parents[2] / "app" / "core" / "Research" / "discovery"
_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_ADAPTER_MODULE_NAME = "tldw_Server_API.app.core.Research.discovery.gateway_adapters"
_FAMILY_MODULE_NAME = "tldw_Server_API.app.core.Research.discovery.biorxiv_medrxiv"
_HTTP_HOP_MODULE = "tldw_Server_API.app.core.Security.http_hop"
_RECORDED_FIXTURES = {
    ("semantic_scholar_v2", "foundation-v2"): ("semantic_scholar_success.json",),
    ("crossref_v2", "foundation-v2"): ("crossref_success.json",),
    ("arxiv_v2", "foundation-v2"): ("arxiv_success.xml",),
    ("pubmed_v2", "foundation-v2"): (
        "pubmed_esearch_success.json",
        "pubmed_esummary_success.json",
    ),
    ("zenodo_v2", "foundation-v2"): ("zenodo_success.json",),
    ("figshare_v2", "foundation-v2"): ("figshare_success.json",),
    ("osf_v2", "foundation-v2"): ("osf_success.json",),
}
_FAMILY_RECORDED_FIXTURES = {
    ("biorxiv_europe_pmc_search_aggregator", "europe_pmc_preprint_v2", "europe-pmc-preprint-v2"): (
        "europe_pmc_biorxiv_success.json",
    ),
    ("medrxiv_europe_pmc_search_aggregator", "europe_pmc_preprint_v2", "europe-pmc-preprint-v2"): (
        "europe_pmc_medrxiv_success.json",
    ),
    ("biorxiv_details_lookup_direct", "biorxiv_details_v2", "biorxiv-details-v2"): (
        "biorxiv_details_doi_success.json",
    ),
    ("medrxiv_details_lookup_direct", "biorxiv_details_v2", "biorxiv-details-v2"): (
        "medrxiv_details_doi_success.json",
    ),
    ("biorxiv_details_interval_direct", "biorxiv_details_v2", "biorxiv-details-v2"): (
        "biorxiv_details_interval_page_1.json",
        "biorxiv_details_interval_page_2.json",
    ),
    ("medrxiv_details_interval_direct", "biorxiv_details_v2", "biorxiv-details-v2"): (
        "biorxiv_details_interval_page_1.json",
        "biorxiv_details_interval_page_2.json",
    ),
}
_V2_ROOT_MODULES = frozenset(
    {
        "contracts.py",
        "registry.py",
        "planner.py",
        "executor.py",
        "gateway_adapters.py",
        "gateway.py",
    }
)
_FAMILY_ROOT_MODULES = frozenset({"biorxiv_medrxiv.py"})
_EXPECTED_LOCAL_CLOSURE = _V2_ROOT_MODULES | {
    "identity.py",
    "catalog.py",
    "models.py",
}
_EXPECTED_FAMILY_LOCAL_CLOSURE = _EXPECTED_LOCAL_CLOSURE | _FAMILY_ROOT_MODULES
_IMPORT_BOOTSTRAP_PATHS = {
    relative_path: _REPO_ROOT / relative_path
    for relative_path in (
        "tldw_Server_API/__init__.py",
        "tldw_Server_API/app/__init__.py",
        "tldw_Server_API/app/core/__init__.py",
        "tldw_Server_API/app/core/testing.py",
        "tldw_Server_API/app/core/Research/__init__.py",
        "tldw_Server_API/app/core/Research/discovery/__init__.py",
        "tldw_Server_API/app/core/Security/__init__.py",
    )
}
_EXPECTED_IMPORT_DIGESTS = {
    "contracts.py": "795b84090cafaa034f2a12af4e6f3b3c6ddbb9ebb3f1607db1ae714cbd8d4ea5",
    "registry.py": "56ffcb107d482f12c2c2477f563b571e716ba73d0937e1a632c89a59b6c1b797",
    "planner.py": "f9b97beef319784481918a64d25ccad69a4d5a784f4f7e646856d70617d8ee9c",
    "executor.py": "adb372ce8329a5fc80ddb30237c08e6f1164911e4e1d0097f3a6122dc7708216",
    "gateway_adapters.py": "ea8c72d2b9fe6ea818249606c2ae6c43a2e50620566e620cf895f89937fc6b95",
    "gateway.py": "d3978650819bcce93a6880d217b6b6688d8c7fd3dd398faef440b03e088062b4",
    "identity.py": "233069fec1e798085a85b14bc1d887a585e22b8a6a6ddaa7dd90001f65b2668d",
    "catalog.py": "2aed2f8efc153fb962668b00c1c3d0f2d51eea78ca03e9c181d30c02d2f7e8e8",
    "models.py": "b3e92240a262c80ac8dc8ab26185ac94565e63dbbd12b5fdfd4b970680263e3c",
    "tldw_Server_API/__init__.py": "b70c123a5edf6dd5edd30b5fd42c2d4c184032b550c3571ba50791de7d61ce63",
    "tldw_Server_API/app/__init__.py": "4d024921beea6dc90cd29b6a2699f05de3e1b428b14b7cac356b1cf83544495a",
    "tldw_Server_API/app/core/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tldw_Server_API/app/core/testing.py": "5d24cb6e3d2c24e2e5978e28489a7cdc536f8ce6c3dd99bcd891ab6127413f16",
    "tldw_Server_API/app/core/Research/__init__.py": "7c52a1d6f02b53490c86e1f4ace834428b1358036ba9f5fb5ec50524dbab7db9",
    "tldw_Server_API/app/core/Research/discovery/__init__.py": "7c52a1d6f02b53490c86e1f4ace834428b1358036ba9f5fb5ec50524dbab7db9",
    "tldw_Server_API/app/core/Security/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
_EXPECTED_AST_DIGESTS = {
    "contracts.py": "f0d47f95cbd919b0834c0426deb32e2569bb0f7867082f16f74515ef5e96f70e",
    "registry.py": "260df2717fcf7cdd91a926dda694bf0517611f0d6378049448581637b4a3def3",
    "planner.py": "dd58f49265b86e44cf1c00746218f42a01ec4b16743d06052fdd42a668df961d",
    "executor.py": "f443e9d69db48379c9675078fb30175168e88da4433d3b842e5387fa9567c2ee",
    "gateway_adapters.py": "9dbcc2dde32f0c4ebede032ae73f68677638e03c53799998af0535ef55d4c656",
    "gateway.py": "f1a55ba9a36fb667a13ccd5426c6e0b5a04894bb5de7a33bbd0f60d7a7e7fc06",
    "identity.py": "f59640d8f793fd3ebd11df49d333f73ebee9e59efb549393116bee2a241f5f06",
    "catalog.py": "6f89e526a0cee3f934fc04da6adc5698800842e9f798ac45e35847d859e08ba6",
    "models.py": "7f6e2b159cf9c42af917a79f24ae2ebd5b4a62c84f81b71766a9d4108f2f57fc",
    "tldw_Server_API/__init__.py": "450c3d7751c7b4d86400f87ae9bbdbdec9306297f44dc3dde652c1508d1bb8df",
    "tldw_Server_API/app/__init__.py": "88280dc29abf20f25212758d3a6497a3be5e5647b78037128817634245ab39b6",
    "tldw_Server_API/app/core/__init__.py": "3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6",
    "tldw_Server_API/app/core/testing.py": "30f64436308026ef99b2415d7f65a265fb4db35f28643adbcbd78ada5c659507",
    "tldw_Server_API/app/core/Research/__init__.py": "73cdb8ab06dbdbca4a15f25e9695273f19a55fce7670872d5374c0177e1302a4",
    "tldw_Server_API/app/core/Research/discovery/__init__.py": "c1c1e9c13c89d64312d5ca46709fc4046bd644cd450460e33b50f380d9dc7622",
    "tldw_Server_API/app/core/Security/__init__.py": "3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6",
}
_EXPECTED_FAMILY_RAW_DIGESTS = {
    "biorxiv_medrxiv.py": "ee77fb9bc5da1cb93dc88baea86c9cd5b6a6e961d02faec97faa66cbcf383af9",
}
_EXPECTED_FAMILY_IMPORT_DIGESTS = {
    "biorxiv_medrxiv.py": "a9a057e486c28731299b0997b04862a6e81dc6454d73d4fc94d0806d6831ebf3",
}
_EXPECTED_FAMILY_AST_DIGESTS = {
    "biorxiv_medrxiv.py": "21f46f0468fc0a83877d6cc4f1bb8830bdc48bbe4c10ee1c1db90d9144e67f95",
}
_EXPECTED_FAMILY_LOCAL_IMPORTS = {
    "contracts": {
        "MAX_PAGINATION_CURSOR",
        "AccessRoute",
        "BackendDefinition",
        "BoundedDecimalQueryValuePolicy",
        "BoundedTextQueryValuePolicy",
        "CredentialRequirement",
        "CredentialStatus",
        "DiscoveryOutcomeIdentity",
        "ExactOrigin",
        "ExactQueryValuePolicy",
        "ExecutionMode",
        "LiteralTermsQueryValuePolicy",
        "OperationKind",
        "PathSlot",
        "PathSlotKind",
        "PathTemplate",
        "PlannedDispatchGroup",
        "PredicateOperator",
        "QueryMode",
        "ReadinessOverlay",
        "ReadinessState",
        "RouteKind",
        "RouteLimits",
        "RoutePolicy",
        "RouteReadiness",
        "SourceConstraint",
        "SourceDefinition",
        "SourcePredicate",
        "SourceRouteReference",
    },
    "executor": {
        "BoundDispatch",
        "DiscoveryAdapter",
        "DiscoveryAdapterError",
        "DiscoveryAdapterResult",
        "DiscoveryCandidate",
        "NumericCursor",
    },
    "gateway_adapters": {
        "MonotonicClock",
        "_ParseDeadlineExceeded",
        "_ParseGuard",
        "_ParseLimitExceeded",
        "_ParsingProfile",
        "_PayloadInvalid",
        "_base_record",
        "_canonical_decimal_text",
        "_checked_response",
        "_optional_text",
        "_raise_adapter_error",
        "_require_dict",
        "_require_list",
        "_required_text",
        "_strict_json",
    },
    "identity": {"build_fingerprint"},
    "registry": {"DiscoveryRegistry", "foundation_readiness", "foundation_registry"},
}
_EXPECTED_GATEWAY_IMPORTS = {
    "executor.py": {
        "DiscoveryGatewayError",
        "DiscoveryGatewayResponse",
        "DiscoveryGatewayTrace",
        "reconstruct_redirect_intent",
    },
    "gateway_adapters.py": {"DiscoveryGatewayResponse"},
    "biorxiv_medrxiv.py": set(),
}
_EXPECTED_IDENTITY_IMPORTS = {
    "executor.py": {"build_fingerprint"},
    "gateway_adapters.py": {
        "build_fingerprint",
        "canonicalize_url",
        "has_unsafe_url_material",
        "normalize_doi",
    },
    "biorxiv_medrxiv.py": {"build_fingerprint"},
}
_EXPECTED_HTTP_HOP_IMPORTS = {
    "HTTPHopError",
    "HTTPHopLimits",
    "HTTPHopResponse",
    "NormalizedHTTPHopRequest",
    "request_http_hop",
}
_EXPECTED_IMPORTED_ATTRIBUTE_PATHS = {
    "contracts.py": {
        "hashlib.sha256",
        "json.dumps",
        "re.compile",
    },
    "registry.py": {
        ".contracts.CredentialRequirement.API_KEY",
        ".contracts.CredentialRequirement.NONE",
        ".contracts.CredentialStatus.NOT_REQUIRED",
        ".contracts.CredentialStatus.OUT_OF_SCOPE",
        ".contracts.QueryMode.STRUCTURED_QUERY",
        ".contracts.ReadinessState.CREDENTIALED_OUT_OF_SCOPE",
        ".contracts.ReadinessState.READY",
        ".contracts.RouteKind.DIRECT",
        ".contracts.SourceConstraint.NATIVE_CORPUS",
    },
    "planner.py": {
        ".contracts.CredentialRequirement.NONE",
        ".contracts.OperationKind.CONDITIONAL_SUMMARY",
        ".contracts.OperationKind.SEARCH",
        ".contracts.PathSlotKind.DATE",
        ".contracts.PathSlotKind.DOI_REGISTRANT",
        ".contracts.PathSlotKind.DOI_SUFFIX",
        ".contracts.PathSlotKind.UINT",
        ".contracts.QueryMode.CATEGORY_BROWSE",
        ".contracts.QueryMode.DATE_INTERVAL",
        ".contracts.QueryMode.GENERAL_FREE_TEXT",
        ".contracts.QueryMode.IDENTIFIER_LOOKUP",
        ".contracts.QueryMode.STRUCTURED_QUERY",
        ".contracts.ReadinessState.READY",
        ".contracts.SkippedCode.CREDENTIALED_OUT_OF_SCOPE",
        ".contracts.SkippedCode.QUERY_MODE_NOT_SUPPORTED",
        ".contracts.SkippedCode.ROUTE_NOT_READY",
        ".contracts.SkippedStatus.SKIPPED",
        ".contracts.SkippedStatus.UNAVAILABLE",
        "datetime.date.fromisoformat",
        "hashlib.sha256",
        "json.dumps",
        "re.compile",
        "unicodedata.category",
        "unicodedata.normalize",
    },
    "executor.py": {
        ".contracts.AttributionMatch.MATCH",
        ".contracts.CredentialRequirement.API_KEY",
        ".contracts.CredentialRequirement.NONE",
        ".contracts.OperationKind.SEARCH",
        ".contracts.PathSlotKind.UINT",
        ".contracts.SkippedCode.CREDENTIALED_OUT_OF_SCOPE",
        ".contracts.SkippedCode.QUERY_MODE_NOT_SUPPORTED",
        ".contracts.SkippedCode.ROUTE_NOT_READY",
        ".contracts.SkippedStatus.SKIPPED",
        ".contracts.SkippedStatus.UNAVAILABLE",
        "asyncio.CancelledError",
        "asyncio.Task",
        "asyncio.TimeoutError",
        "asyncio.create_task",
        "asyncio.current_task",
        "asyncio.wait",
        "copy.deepcopy",
        "email.utils.format_datetime",
        "email.utils.parsedate_to_datetime",
        "math.isfinite",
        "re.compile",
        "time.monotonic",
        "uuid.uuid4",
        "weakref.WeakKeyDictionary",
    },
    "gateway_adapters.py": {
        ".contracts.DiscoveryOutcomeIdentity.from_fingerprint",
        ".contracts.OperationKind.CONDITIONAL_SUMMARY",
        ".contracts.OperationKind.SEARCH",
        "ipaddress.ip_address",
        "json.JSONDecodeError",
        "json.loads",
        "math.isfinite",
        "re.ASCII",
        "re.IGNORECASE",
        "re.compile",
        "time.monotonic",
        "xml.etree.ElementTree.Element",
        "xml.etree.ElementTree.ParseError",
        "xml.etree.ElementTree.TreeBuilder",
    },
    "gateway.py": {
        "datetime.date.fromisoformat",
        "ipaddress.IPv4Address",
        "ipaddress.IPv6Address",
        "ipaddress.ip_address",
        "ipaddress.ip_network",
        "json.dumps",
        "re.compile",
        "time.monotonic",
        "tldw_Server_API.app.core.Research.discovery.contracts.CredentialRequirement.NONE",
        "tldw_Server_API.app.core.Research.discovery.contracts.PathSlotKind.DATE",
        "tldw_Server_API.app.core.Research.discovery.contracts.PathSlotKind.DOI_REGISTRANT",
        "tldw_Server_API.app.core.Research.discovery.contracts.PathSlotKind.DOI_SUFFIX",
        "tldw_Server_API.app.core.Research.discovery.contracts.PathSlotKind.UINT",
        "unicodedata.normalize",
    },
    "identity.py": {
        "hashlib.sha256",
        "json.dumps",
        "re.IGNORECASE",
        "re.compile",
        "re.sub",
    },
    "biorxiv_medrxiv.py": {
        ".contracts.CredentialRequirement.NONE",
        ".contracts.CredentialStatus.NOT_REQUIRED",
        ".contracts.DiscoveryOutcomeIdentity.from_fingerprint",
        ".contracts.OperationKind.SEARCH",
        ".contracts.PathSlotKind.DATE",
        ".contracts.PathSlotKind.DOI_REGISTRANT",
        ".contracts.PathSlotKind.DOI_SUFFIX",
        ".contracts.PathSlotKind.UINT",
        ".contracts.PredicateOperator.EQUALS_ANY",
        ".contracts.QueryMode.CATEGORY_BROWSE",
        ".contracts.QueryMode.DATE_INTERVAL",
        ".contracts.QueryMode.GENERAL_FREE_TEXT",
        ".contracts.QueryMode.IDENTIFIER_LOOKUP",
        ".contracts.ReadinessState.READY",
        ".contracts.RouteKind.AGGREGATOR",
        ".contracts.RouteKind.DIRECT",
        ".contracts.SourceConstraint.NATIVE_CORPUS",
        ".contracts.SourceConstraint.PROVIDER_SOURCE_FILTER",
        "datetime.date.fromisoformat",
        "re.ASCII",
        "re.compile",
        "time.monotonic",
        "unicodedata.normalize",
    },
    "catalog.py": set(),
    "models.py": set(),
}
_ALLOWED_DYNAMIC_GETATTR_TARGETS = {
    "contracts.py": {"self"},
    "registry.py": {"self"},
    "executor.py": {"copied", "value"},
    "gateway.py": {"limits"},
}
_FORBIDDEN_DYNAMIC_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "import_module",
    "locals",
    "open",
    "vars",
}
_FORBIDDEN_EFFECT_ATTRIBUTES = {
    "__class__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__subclasses__",
    "HTTPConnection",
    "HTTPSConnection",
    "ClientSession",
    "CookieJar",
    "PoolManager",
    "Session",
    "afetch",
    "afetch_json",
    "browser",
    "chromium",
    "cookiejar",
    "cookies",
    "create_connection",
    "create_datagram_endpoint",
    "create_server",
    "create_subprocess_exec",
    "create_subprocess_shell",
    "create_unix_connection",
    "environ",
    "fetch",
    "fetch_json",
    "firefox",
    "get_cookie",
    "getenv",
    "getaddrinfo",
    "gethostbyaddr",
    "gethostbyname",
    "gethostbyname_ex",
    "getnameinfo",
    "goto",
    "keyring",
    "launch",
    "modules",
    "new_context",
    "new_page",
    "open",
    "open_connection",
    "open_unix_connection",
    "popen",
    "putenv",
    "read_bytes",
    "read_text",
    "request_http_hop",
    "run",
    "set_cookie",
    "sleep",
    "sock_connect",
    "sock_accept",
    "sock_recv",
    "sock_recv_into",
    "sock_recvfrom",
    "sock_recvfrom_into",
    "sock_sendall",
    "sock_sendfile",
    "sock_sendto",
    "start_server",
    "start_unix_server",
    "system",
    "unsetenv",
    "urlopen",
    "webkit",
}
_ALLOWED_ASYNCIO_ATTRIBUTES = {
    "CancelledError",
    "Task",
    "TimeoutError",
    "create_task",
    "current_task",
    "wait",
}
_FAMILY_FORBIDDEN_IMPORT_FRAGMENTS = (
    "aiohttp",
    "authnz",
    "browser",
    "config",
    "cookie",
    "credential",
    "db_management",
    "http.client",
    "http_client",
    "httpx",
    "ingestion_media_processing",
    "keyring",
    "playwright",
    "requests",
    "security.http_hop",
    "selenium",
    "socket",
    "subprocess",
    "third_party",
    "urllib.request",
    "urllib3",
    "web_scraping",
)
_FAMILY_FORBIDDEN_EXACT_IMPORTS = {
    ".gateway",
    ".oa",
    "tldw_Server_API.app.core.Media",
    "tldw_Server_API.app.core.Research.discovery.gateway",
    "tldw_Server_API.app.core.Research.discovery.oa",
}


def _resolve_import_path(node: ast.AST, bindings: dict[str, str]) -> str | None:
    """Resolve a name/attribute expression rooted in one imported binding."""
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in bindings:
        return None
    return ".".join((bindings[node.id], *reversed(attributes)))


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    """Map imported names and simple aliases to their fully qualified paths."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                bindings[local_name] = alias.name if alias.asname else local_name
        elif isinstance(node, ast.ImportFrom):
            prefix = f"{'.' * node.level}{node.module or ''}"
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{prefix}.{alias.name}"

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value = None
            targets: tuple[ast.expr, ...] = ()
            if isinstance(node, ast.Assign):
                value = node.value
                targets = tuple(node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                targets = (node.target,)
            if value is None:
                continue
            resolved = _resolve_import_path(value, bindings)
            if resolved is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in bindings:
                    bindings[target.id] = resolved
                    changed = True
    return bindings


def _adapter_module():
    return importlib.import_module(_ADAPTER_MODULE_NAME)


def _family_module():
    return importlib.import_module(_FAMILY_MODULE_NAME)


def _canonical_import_digest(tree: ast.AST) -> str:
    """Hash exact imports and imported symbols while intentionally ignoring aliases."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(f"I:0:{alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(f"F:{node.level}:{node.module or ''}:{alias.name}" for alias in node.names)
    return hashlib.sha256("\n".join(sorted(imports)).encode()).hexdigest()


def _semantic_ast_digest(tree: ast.AST) -> str:
    """Hash executable syntax while ignoring formatting, comments, and locations."""
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _frozen_digest_violations(
    tree: ast.AST,
    module_key: str,
    *,
    check_import_digest: bool = True,
) -> list[str]:
    """Return drift from one reviewed import and executable-syntax snapshot."""
    violations = []
    expected_import_digest = _EXPECTED_FAMILY_IMPORT_DIGESTS.get(
        module_key,
        _EXPECTED_IMPORT_DIGESTS.get(module_key),
    )
    expected_ast_digest = _EXPECTED_FAMILY_AST_DIGESTS.get(
        module_key,
        _EXPECTED_AST_DIGESTS.get(module_key),
    )
    if expected_import_digest is None or expected_ast_digest is None:
        raise KeyError(module_key)
    import_digest = _canonical_import_digest(tree)
    if check_import_digest and import_digest != expected_import_digest:
        violations.append(f"{module_key}:import_digest:{import_digest}")
    ast_digest = _semantic_ast_digest(tree)
    if ast_digest != expected_ast_digest:
        violations.append(f"{module_key}:ast_digest:{ast_digest}")
    return violations


def _boundary_violations(source: str, filename: str, *, check_import_digest: bool = True) -> list[str]:
    """Return direct import and effect seams outside one file's frozen allowlist."""
    tree = ast.parse(source, filename=filename)
    violations = []
    violations.extend(
        _frozen_digest_violations(
            tree,
            filename,
            check_import_digest=check_import_digest,
        )
    )
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    import_bindings = _import_bindings(tree)
    asyncio_aliases = {name for name, qualified_path in import_bindings.items() if qualified_path == "asyncio"}
    if filename == "biorxiv_medrxiv.py":
        local_imports = {
            node.module: {alias.name for alias in node.names}
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        if local_imports != _EXPECTED_FAMILY_LOCAL_IMPORTS:
            violations.append(f"{filename}:family_local_imports:{local_imports}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "*" for alias in node.names):
                violations.append(f"{filename}:{node.lineno}:star_import")
            if filename == "biorxiv_medrxiv.py":
                imported_modules = (
                    tuple(alias.name for alias in node.names)
                    if isinstance(node, ast.Import)
                    else (f"{'.' * node.level}{node.module or ''}",)
                )
                for imported_module in imported_modules:
                    folded_module = imported_module.casefold()
                    if imported_module in _FAMILY_FORBIDDEN_EXACT_IMPORTS or any(
                        fragment in folded_module for fragment in _FAMILY_FORBIDDEN_IMPORT_FRAGMENTS
                    ):
                        violations.append(f"{filename}:{node.lineno}:forbidden_import:{imported_module}")
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "gateway":
                imported = {alias.name for alias in node.names}
                if imported != _EXPECTED_GATEWAY_IMPORTS.get(filename, set()):
                    violations.append(f"{filename}:{node.lineno}:gateway_symbols:{sorted(imported)}")
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "identity":
                imported = {alias.name for alias in node.names}
                if imported != _EXPECTED_IDENTITY_IMPORTS.get(filename, set()):
                    violations.append(f"{filename}:{node.lineno}:identity_symbols:{sorted(imported)}")
            if isinstance(node, ast.ImportFrom) and node.module == _HTTP_HOP_MODULE:
                imported = {alias.name for alias in node.names}
                if filename != "gateway.py" or imported != _EXPECTED_HTTP_HOP_IMPORTS:
                    violations.append(f"{filename}:{node.lineno}:http_hop_symbols:{sorted(imported)}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_DYNAMIC_CALLS:
                violations.append(f"{filename}:{node.lineno}:call:{node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_EFFECT_ATTRIBUTES:
                if not (filename == "gateway.py" and node.func.attr == "request_http_hop"):
                    violations.append(f"{filename}:{node.lineno}:call:{node.func.attr}")
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                target_name = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
                if target_name not in _ALLOWED_DYNAMIC_GETATTR_TARGETS.get(filename, set()):
                    attribute_name = (
                        node.args[1].value
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                        else "computed"
                    )
                    violations.append(f"{filename}:{node.lineno}:dynamic_getattr:{attribute_name}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_EFFECT_ATTRIBUTES:
            if not (filename == "gateway.py" and node.attr == "request_http_hop"):
                violations.append(f"{filename}:{node.lineno}:attribute:{node.attr}")
        if isinstance(node, ast.Attribute) and not isinstance(parents.get(node), ast.Attribute):
            imported_path = _resolve_import_path(node, import_bindings)
            if imported_path is not None and imported_path not in _EXPECTED_IMPORTED_ATTRIBUTE_PATHS[filename]:
                violations.append(f"{filename}:{node.lineno}:imported_attribute:{imported_path}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in asyncio_aliases
            and node.attr not in _ALLOWED_ASYNCIO_ATTRIBUTES
        ):
            violations.append(f"{filename}:{node.lineno}:asyncio_attribute:{node.attr}")
        if isinstance(node, ast.Name) and node.id == "request_http_hop" and filename != "gateway.py":
            violations.append(f"{filename}:{node.lineno}:name:request_http_hop")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_DYNAMIC_CALLS:
            violations.append(f"{filename}:{node.lineno}:name:{node.id}")
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            violations.append(f"{filename}:{node.lineno}:name:__builtins__")
    return violations


def _local_dependency_closure(roots: frozenset[str]) -> frozenset[str]:
    """Resolve direct local discovery imports without importing their modules."""
    discovered = set(roots)
    pending = list(roots)
    absolute_prefix = "tldw_Server_API.app.core.Research.discovery."
    while pending:
        filename = pending.pop()
        tree = ast.parse((_DISCOVERY_ROOT / filename).read_text(encoding="utf-8"), filename=filename)
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            local_names: tuple[str, ...] = ()
            if node.level == 1:
                local_names = (node.module.split(".", 1)[0],) if node.module else tuple(a.name for a in node.names)
            elif node.level == 0 and node.module and node.module.startswith(absolute_prefix):
                local_names = (node.module.removeprefix(absolute_prefix).split(".", 1)[0],)
            for local_name in local_names:
                candidate = f"{local_name}.py"
                if (_DISCOVERY_ROOT / candidate).is_file() and candidate not in discovered:
                    discovered.add(candidate)
                    pending.append(candidate)
    return frozenset(discovered)


def _foundation_plan(
    *,
    source_ids: tuple[str, ...] | None = None,
    result_limit: int = 100,
    budget: BudgetCeilings | None = None,
):
    registry = foundation_registry()
    requested = source_ids or tuple(source.catalog_source_id for source in registry.sources)
    plan = compile_discovery_plan(
        PlanningRequest(requested, "bounded discovery", (), result_limit),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=budget or BudgetCeilings(7, 8, 1, 0, 0, 1_000_000, result_limit),
    )
    return registry, plan


def _family_plan(route_id: str):
    module = _family_module()
    source_id = route_id.split("_", 1)[0]
    if route_id.endswith("europe_pmc_search_aggregator"):
        query = GeneralFreeTextQuery("bounded family discovery")
        result_limit = 100
        max_pages = max_dispatches = 1
    elif route_id.endswith("details_lookup_direct"):
        query = IdentifierLookupQuery(f"10.5555/{source_id}.details.synthetic")
        result_limit = 30
        max_pages = max_dispatches = 1
    else:
        query = DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience")
        result_limit = 120
        max_pages = max_dispatches = 4
    registry = module.biorxiv_medrxiv_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest((source_id,), query, (), result_limit),
        registry=registry,
        readiness=module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=BudgetCeilings(
            max_route_attempts=1,
            max_physical_dispatches=max_dispatches,
            max_pages_per_route=max_pages,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=20_000 * max_dispatches,
            max_results=result_limit,
        ),
    )
    return registry, plan


def _derived_medrxiv_interval_body(body: bytes) -> bytes:
    payload = json.loads(body)
    for item in payload["collection"]:
        item["server"] = "medRxiv"
        item["doi"] = item["doi"].replace("biorxiv", "medrxiv")
        item["title"] = item["title"].replace("interval", "medRxiv interval")
        item["abstract"] = item["abstract"].replace("interval", "medRxiv interval")
        item["authors"] = item["authors"].replace("Example", "MedExample")
        if item["published"] != "NA":
            item["published"] = item["published"].replace("interval", "medrxiv.interval")
    return json.dumps(payload, separators=(",", ":")).encode()


def _family_fixture_bodies() -> dict[tuple[str, str, str], tuple[bytes, ...]]:
    bodies = {
        identity: tuple((_FIXTURE_ROOT / filename).read_bytes() for filename in filenames)
        for identity, filenames in _FAMILY_RECORDED_FIXTURES.items()
    }
    medrxiv_interval = next(identity for identity in bodies if identity[0] == "medrxiv_details_interval_direct")
    bodies[medrxiv_interval] = tuple(_derived_medrxiv_interval_body(body) for body in bodies[medrxiv_interval])
    return bodies


def _family_boundary_route_ids(registry, readiness) -> tuple[set[str], set[str], set[str]]:
    foundation_route_ids = {route.route_id for route in foundation_registry().routes}
    registry_route_ids = {route.route_id for route in registry.routes if route.route_id not in foundation_route_ids}
    readiness_route_ids = {entry.route_id for entry in readiness.routes if entry.route_id not in foundation_route_ids}
    ready_route_ids = {
        entry.route_id
        for entry in readiness.routes
        if entry.state is ReadinessState.READY and entry.route_id not in foundation_route_ids
    }
    return registry_route_ids, readiness_route_ids, ready_route_ids


def _response(route, intent, body: bytes, *, status_code: int = 200) -> DiscoveryGatewayResponse:
    content_type = "application/atom+xml; charset=utf-8" if route.adapter_id == "arxiv_v2" else "application/json"
    origin = route.policy.origin
    return DiscoveryGatewayResponse(
        status_code=status_code,
        headers=(("content-type", content_type),),
        body=body,
        trace=DiscoveryGatewayTrace(
            route_id=route.route_id,
            policy_digest=route.policy.policy_digest,
            scheme=origin.scheme,
            requested_host=origin.host,
            tls_server_name=origin.host,
            port=origin.port,
            method=intent.method,
            path=intent.path,
            query_keys=tuple(pair.name for pair in intent.query_pairs),
            timeout_ms=intent.limits.timeout_ms,
            max_response_bytes=intent.limits.max_response_bytes,
            http_limits=HTTPHopLimits(),
            status_code=status_code,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=len(body),
            decoded_bytes=len(body),
            elapsed_ms=1,
        ),
        redirect_location=None,
        retry_after=None,
    )


def _install_runtime_tripwires(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject alternate egress plus late imports of deferred retrieval systems."""
    from tldw_Server_API.app.core.Security import http_hop

    def forbidden(*_args, **_kwargs):
        raise AssertionError("discovery V2 attempted an alternate effect path")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for name in ("getaddrinfo", "gethostbyaddr", "gethostbyname", "gethostbyname_ex", "getnameinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http_hop, "request_http_hop", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    for name in (
        "open_connection",
        "start_server",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "open_unix_connection",
        "start_unix_server",
    ):
        monkeypatch.setattr(asyncio, name, forbidden)
    for name in ("Popen", "call", "check_call", "check_output", "run"):
        monkeypatch.setattr(subprocess, name, forbidden)

    original_import = builtins.__import__
    blocked_import_fragments = (
        "aiohttp",
        "browser",
        "cookie",
        "credential",
        "http.client",
        "httpx",
        "keyring",
        "playwright",
        "requests",
        "selenium",
        "socket",
        "subprocess",
        "tldw_server_api.app.core.authnz",
        "tldw_server_api.app.core.config",
        "tldw_server_api.app.core.db_management",
        "tldw_server_api.app.core.http_client",
        "tldw_server_api.app.core.ingestion_media_processing",
        "tldw_server_api.app.core.media",
        "tldw_server_api.app.core.research.discovery.oa",
        "tldw_server_api.app.core.security.http_hop",
        "tldw_server_api.app.core.third_party",
        "tldw_server_api.app.core.web_scraping",
        "urllib.request",
        "urllib3",
    )

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute_name = name
        package = globals.get("__package__") if level and globals else None
        if level and package:
            absolute_name = importlib.util.resolve_name(f"{'.' * level}{name}", package)
        if any(fragment in absolute_name.casefold() for fragment in blocked_import_fragments):
            raise AssertionError(f"discovery V2 imported deferred effect system: {absolute_name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _aggregator_plan():
    base = foundation_registry()
    original = base.get_source("semantic_scholar")
    route_id = original.route_references[0].route_id
    first_predicate = SourcePredicate(
        ("source", "collection"),
        PredicateOperator.EQUALS_ANY,
        ("shared",),
    )
    second_predicate = SourcePredicate(
        ("source", "collection"),
        PredicateOperator.EQUALS_ANY,
        ("other",),
    )
    first_source = replace(
        original,
        catalog_source_id="target_a",
        display_name="Target A",
        aliases=(),
        route_references=(SourceRouteReference(route_id, first_predicate),),
    )
    second_source = replace(
        first_source,
        catalog_source_id="target_b",
        display_name="Target B",
        priority=first_source.priority + 1,
        route_references=(SourceRouteReference(route_id, second_predicate),),
    )
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version="synthetic-aggregator-v1",
        sources=tuple(first_source if source is original else source for source in base.sources) + (second_source,),
        routes=tuple(
            (
                replace(
                    route,
                    route_kind=RouteKind.AGGREGATOR,
                    adapter_id="shared_aggregator_v2",
                    adapter_version="synthetic-v1",
                    source_constraint=SourceConstraint.PROVIDER_SOURCE_FILTER,
                    attribution_basis="source.collection",
                )
                if route.route_id == route_id
                else route
            )
            for route in base.routes
        ),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest(("target_a", "target_b"), "coalesced", (), 10),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(2, 1, 1, 0, 0, 20_000, 10),
    )
    return registry, plan


def test_enabled_registry_factory_profiles_recordings_and_plan_are_exactly_equal() -> None:
    registry, plan = _foundation_plan()
    readiness = foundation_readiness(ExecutionMode.OFFLINE_FIXTURE)
    ready_route_ids = {entry.route_id for entry in readiness.routes if entry.state is ReadinessState.READY}
    ready_routes = tuple(route for route in registry.routes if route.route_id in ready_route_ids)
    registry_identities = {(route.adapter_id, route.adapter_version) for route in ready_routes}
    plan_identities = {(group.adapter_id, group.adapter_version) for group in plan.dispatch_groups}
    module = _adapter_module()
    factory = module.foundation_gateway_adapters()

    assert registry_identities == plan_identities == set(_RECORDED_FIXTURES) == set(module._PARSING_PROFILES)
    assert set(factory) == {adapter_id for adapter_id, _version in registry_identities}
    assert all(route.credential_requirement is CredentialRequirement.NONE for route in ready_routes)
    assert {adapter.__module__ for adapter in factory.values()} == {_ADAPTER_MODULE_NAME}
    assert "openalex_v2" not in factory
    assert "openalex_v2" not in {adapter_id for adapter_id, _version in module._PARSING_PROFILES}
    assert [skipped.requested_source_id for skipped in plan.skipped] == ["openalex"]


@pytest.mark.parametrize(
    ("filename", "probe"),
    (
        ("contracts.py", "import importlib\nimportlib.import_module('requests')"),
        ("contracts.py", "import tldw_Server_API.app.core as core\ncore.http_client.fetch('https://example.test')"),
        ("executor.py", "from asyncio import open_connection as connect\nconnect('example.test', 443)"),
        ("executor.py", "from asyncio import create_subprocess_exec as launch\nlaunch('curl')"),
        ("executor.py", "from time import sleep as pause\npause(1)"),
        ("executor.py", "from . import gateway as g\ng.request_http_hop(None)"),
        ("gateway_adapters.py", "from urllib import request as parser"),
        ("gateway.py", "from tldw_Server_API.app.core.Security import http_hop"),
        ("executor.py", "import sys\nsys.modules['requests']"),
    ),
)
def test_boundary_scanner_rejects_exact_import_drift(filename: str, probe: str) -> None:
    source = (_DISCOVERY_ROOT / filename).read_text(encoding="utf-8") + f"\n{probe}\n"

    assert any("import_digest" in violation for violation in _boundary_violations(source, filename))


@pytest.mark.parametrize(
    ("filename", "probe", "expected_marker"),
    (
        ("executor.py", "asyncio.open_connection('example.test', 443)", "open_connection"),
        ("executor.py", "asyncio.create_subprocess_exec('curl')", "create_subprocess_exec"),
        (
            "executor.py",
            "asyncio.get_running_loop().getaddrinfo('example.test', 443)",
            "getaddrinfo",
        ),
        (
            "executor.py",
            "asyncio.current_task().get_loop().sock_sendto(None, b'', ('example.test', 443))",
            "sock_sendto",
        ),
        (
            "executor.py",
            "import asyncio as eventing\neventing.get_running_loop()",
            "asyncio_attribute:get_running_loop",
        ),
        (
            "executor.py",
            "name = 'open_' + 'connection'\ngetattr(asyncio, name)('example.test', 443)",
            "dynamic_getattr:computed",
        ),
        (
            "executor.py",
            "name = 'sl' + 'eep'\ngetattr(time, name)(1)",
            "dynamic_getattr:computed",
        ),
        (
            "executor.py",
            "time.__dict__['sleep'](1)",
            "attribute:__dict__",
        ),
        (
            "executor.py",
            "email.utils.socket.socket().connect(('example.test', 443))",
            "imported_attribute:email.utils.socket.socket",
        ),
        (
            "executor.py",
            "email.utils.os.getenvb(b'OPENAI_API_KEY')",
            "imported_attribute:email.utils.os.getenvb",
        ),
        (
            "executor.py",
            "eventing = asyncio\neventing.events.socket.socket().connect(('example.test', 443))",
            "imported_attribute:asyncio.events.socket.socket",
        ),
        (
            "executor.py",
            "[asyncio][0].events.socket.socket().connect(('example.test', 443))",
            "ast_digest",
        ),
        (
            "executor.py",
            "(eventing := asyncio)\neventing.events.socket.socket().connect(('example.test', 443))",
            "ast_digest",
        ),
        (
            "executor.py",
            "value = (lambda item: item)(asyncio)\nname = 'open_' + 'connection'\ngetattr(value, name)('example.test', 443)",
            "ast_digest",
        ),
        ("executor.py", "getattr(time, 'sleep')(1)", "dynamic_getattr:sleep"),
        ("contracts.py", "reader = open\nreader('/tmp/secret')", "name:open"),
        (
            "contracts.py",
            "getattr(__builtins__, '__import__')('requests')",
            "dynamic_getattr:__import__",
        ),
    ),
)
def test_boundary_scanner_rejects_effect_aliases_without_import_drift(
    filename: str,
    probe: str,
    expected_marker: str,
) -> None:
    source = (_DISCOVERY_ROOT / filename).read_text(encoding="utf-8") + f"\n{probe}\n"

    violations = _boundary_violations(source, filename, check_import_digest=False)

    assert any(expected_marker in violation for violation in violations), violations


def test_v2_import_closure_has_exact_safe_dependencies_and_one_transport_consumer() -> None:
    closure = _local_dependency_closure(_V2_ROOT_MODULES)

    assert closure == _EXPECTED_LOCAL_CLOSURE
    violations = []
    http_hop_consumers = []
    for filename in sorted(closure):
        source = (_DISCOVERY_ROOT / filename).read_text(encoding="utf-8")
        violations.extend(_boundary_violations(source, filename))
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == _HTTP_HOP_MODULE:
                http_hop_consumers.append(filename)

    assert violations == []
    assert http_hop_consumers == ["gateway.py"]


def test_import_bootstrap_chain_is_frozen_before_runtime_tripwires() -> None:
    expected_paths = {
        "tldw_Server_API/__init__.py",
        "tldw_Server_API/app/__init__.py",
        "tldw_Server_API/app/core/__init__.py",
        "tldw_Server_API/app/core/testing.py",
        "tldw_Server_API/app/core/Research/__init__.py",
        "tldw_Server_API/app/core/Research/discovery/__init__.py",
        "tldw_Server_API/app/core/Security/__init__.py",
    }

    assert set(_IMPORT_BOOTSTRAP_PATHS) == expected_paths
    violations = []
    for module_key, path in _IMPORT_BOOTSTRAP_PATHS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module_key)
        violations.extend(_frozen_digest_violations(tree, module_key))
    assert violations == []

    discovery_init = "tldw_Server_API/app/core/Research/discovery/__init__.py"
    mutated_source = _IMPORT_BOOTSTRAP_PATHS[discovery_init].read_text(encoding="utf-8")
    mutated_source += "\nimport socket\nsocket.create_connection(('example.test', 443))\n"
    mutated_tree = ast.parse(mutated_source, filename=discovery_init)
    mutation_violations = _frozen_digest_violations(mutated_tree, discovery_init)
    assert any("import_digest" in violation for violation in mutation_violations)
    assert any("ast_digest" in violation for violation in mutation_violations)


@pytest.mark.asyncio
async def test_all_ready_registry_adapters_execute_recorded_fixtures_with_only_accounted_gateway_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, plan = _foundation_plan()
    module = _adapter_module()
    adapters = dict(module.foundation_gateway_adapters(monotonic_clock=lambda: 0.0))
    fixture_queues = {
        identity[0]: [(_FIXTURE_ROOT / filename).read_bytes() for filename in filenames]
        for identity, filenames in _RECORDED_FIXTURES.items()
    }
    _install_runtime_tripwires(monkeypatch)
    gateway_calls = []

    async def forbidden_unregistered_adapter(*_args, **_kwargs):
        raise AssertionError("executor invoked an unregistered adapter")

    adapters["unregistered_v2"] = forbidden_unregistered_adapter

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        gateway_calls.append((route.adapter_id, intent.path))
        return _response(route, intent, fixture_queues[route.adapter_id].pop(0))

    expected_dispatches_by_adapter = Counter(
        {adapter_id: len(filenames) for (adapter_id, _version), filenames in _RECORDED_FIXTURES.items()}
    )
    expected_dispatches = expected_dispatches_by_adapter.total()
    dispatch_ids = iter(f"dispatch-{index}" for index in range(1, expected_dispatches + 1))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=adapters,
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
        monotonic_clock=lambda: 0.0,
    )

    remaining_fixture_counts = {adapter_id: len(remaining) for adapter_id, remaining in fixture_queues.items()}
    assert remaining_fixture_counts == dict.fromkeys(fixture_queues, 0), (
        remaining_fixture_counts,
        tuple((outcome.state, outcome.code) for outcome in result.logical_outcomes),
        gateway_calls,
    )
    assert expected_dispatches == 8
    assert len(gateway_calls) == expected_dispatches
    assert Counter(adapter_id for adapter_id, _path in gateway_calls) == expected_dispatches_by_adapter
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (LogicalOutcomeState.SUCCEEDED,) * 7
    assert tuple(candidate.record["provider"] for candidate in result.candidates) == (
        "semantic_scholar",
        "pubmed",
    )
    shared_doi_candidates = [
        candidate for candidate in result.candidates if candidate.record["doi"] == "10.5555/shared.discovery.2026"
    ]
    assert len(shared_doi_candidates) == 1
    assert tuple(contribution.record["provider"] for contribution in shared_doi_candidates[0].contributions) == (
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    )
    assert shared_doi_candidates[0].catalog_source_ids == (
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    )
    assert all(candidate.record["url"].startswith("https://") for candidate in result.candidates)
    assert len(result.usage.physical_records) == expected_dispatches
    assert result.usage.accounting.debited == expected_dispatches
    assert result.truncated_candidates == 0


@pytest.mark.asyncio
async def test_provider_failure_is_typed_and_cannot_fallback_or_dereference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, plan = _foundation_plan(source_ids=("semantic_scholar",), result_limit=1)
    module = _adapter_module()
    adapters = dict(module.foundation_gateway_adapters(monotonic_clock=lambda: 0.0))
    _install_runtime_tripwires(monkeypatch)
    calls = []

    async def forbidden_unregistered_adapter(*_args, **_kwargs):
        raise AssertionError("provider failure invoked fallback adapter")

    adapters["unregistered_v2"] = forbidden_unregistered_adapter

    async def gateway(route, intent, *, is_policy_active):
        calls.append((route.route_id, intent.path))
        return _response(route, intent, b'{"error":"unsafe-fixture-detail"}', status_code=429)

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=adapters,
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-rate-limited",
        monotonic_clock=lambda: 0.0,
    )

    assert len(calls) == 1
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_rate_limited"
    assert "unsafe-fixture-detail" not in repr(result)
    assert result.usage.accounting.debited == 1


@pytest.mark.asyncio
async def test_openalex_real_factory_path_is_typed_skipped_with_zero_runtime_effects() -> None:
    registry, plan = _foundation_plan(
        source_ids=("openalex",),
        result_limit=1,
        budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
    )
    adapters = _adapter_module().foundation_gateway_adapters(monotonic_clock=lambda: 0.0)
    journal = AttemptJournal(physical_ceiling=0)
    calls = []

    async def forbidden(*_args, **_kwargs):
        calls.append("effect")
        raise AssertionError("OpenAlex out-of-scope route performed runtime work")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=adapters,
        gateway=forbidden,
        policy_is_active=lambda _route_id, _digest: calls.append("policy") or True,
        dispatch_id_factory=lambda: calls.append("id") or "impossible",
        journal=journal,
    )

    assert plan.dispatch_groups == ()
    assert plan.allowance.route_attempts == 0
    assert plan.allowance.physical_dispatches == 0
    assert len(result.skipped) == 1
    assert result.skipped[0].code is SkippedCode.CREDENTIALED_OUT_OF_SCOPE
    assert result.candidates == ()
    assert result.logical_outcomes == ()
    assert journal.records == ()
    assert calls == []


@pytest.mark.asyncio
async def test_synthetic_aggregator_distinguishes_match_nonmatch_and_ambiguity_in_one_dispatch() -> None:
    registry, plan = _aggregator_plan()
    group = plan.dispatch_groups[0]
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append((route.route_id, intent.path))
        return _response(route, intent, b"{}")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(
            candidates=(
                DiscoveryCandidate("match-a", {"source": {"collection": "shared"}}),
                DiscoveryCandidate("match-b", {"source": {"collection": "other"}}),
                DiscoveryCandidate("nonmatch", {"source": {"collection": "neither"}}),
                DiscoveryCandidate("ambiguous", {"source": {}}),
            )
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-shared-aggregator",
    )

    assert gateway_calls == [(group.route_id, group.intents[0].path)]
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("match-a", "match-b")
    assert tuple(candidate.catalog_source_ids for candidate in result.candidates) == (("target_a",), ("target_b",))
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.SUCCEEDED,
    )
    assert len(result.usage.physical_records) == 1
    assert result.usage.accounting.debited == 1


@pytest.mark.asyncio
async def test_ambiguous_only_aggregator_result_is_unattributed_valid_empty() -> None:
    registry, plan = _aggregator_plan()
    group = plan.dispatch_groups[0]

    async def gateway(route, intent, *, is_policy_active):
        return _response(route, intent, b"{}")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("ambiguous", {"source": {}}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-ambiguous-aggregator",
    )

    assert result.candidates == ()
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.VALID_EMPTY,
        LogicalOutcomeState.VALID_EMPTY,
    )
    assert len(result.usage.physical_records) == 1
    assert result.usage.accounting.debited == 1


def test_biorxiv_medrxiv_family_has_a_separate_closed_boundary_contract() -> None:
    assert frozenset({"biorxiv_medrxiv.py"}) == _FAMILY_ROOT_MODULES
    assert _local_dependency_closure(_FAMILY_ROOT_MODULES) == _EXPECTED_FAMILY_LOCAL_CLOSURE
    assert set(_EXPECTED_FAMILY_RAW_DIGESTS) == _FAMILY_ROOT_MODULES
    assert set(_EXPECTED_FAMILY_IMPORT_DIGESTS) == _FAMILY_ROOT_MODULES
    assert set(_EXPECTED_FAMILY_AST_DIGESTS) == _FAMILY_ROOT_MODULES

    violations = []
    http_hop_consumers = []
    for filename in sorted(_EXPECTED_FAMILY_LOCAL_CLOSURE):
        module_path = _DISCOVERY_ROOT / filename
        raw_source = module_path.read_bytes()
        if filename in _EXPECTED_FAMILY_RAW_DIGESTS:
            assert hashlib.sha256(raw_source).hexdigest() == _EXPECTED_FAMILY_RAW_DIGESTS[filename]
        source = raw_source.decode("utf-8")
        violations.extend(_boundary_violations(source, filename))
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == _HTTP_HOP_MODULE:
                http_hop_consumers.append(filename)

    assert violations == []
    assert http_hop_consumers == ["gateway.py"]


@pytest.mark.parametrize(
    "probe",
    (
        "import socket\nsocket.create_connection(('example.test', 443))",
        "import http.client\nhttp.client.HTTPSConnection('example.test')",
        "import urllib.request\nurllib.request.urlopen('https://example.test')",
        "from .gateway import dispatch_once\ndispatch_once(None)",
        "import subprocess\nsubprocess.run(['curl', 'https://example.test'])",
        "import webbrowser\nwebbrowser.open('https://example.test')",
        "import http.cookiejar\nhttp.cookiejar.CookieJar()",
        "import keyring\nkeyring.get_password('service', 'user')",
        "import tldw_Server_API.app.core.config",
        "import tldw_Server_API.app.core.AuthNZ",
        "import tldw_Server_API.app.core.DB_Management",
        "import tldw_Server_API.app.core.Web_Scraping",
        "import tldw_Server_API.app.core.Media",
        "import tldw_Server_API.app.core.Research.discovery.oa",
        "import tldw_Server_API.app.core.Security.http_hop",
        "import tldw_Server_API.app.core.Third_Party.BioRxiv",
        "import urllib.request\nurllib.request.urlopen(record['url'])",
    ),
)
def test_family_boundary_scanner_rejects_every_deferred_effect_class(probe: str) -> None:
    filename = "biorxiv_medrxiv.py"
    source = (_DISCOVERY_ROOT / filename).read_text(encoding="utf-8") + f"\n{probe}\n"

    violations = _boundary_violations(source, filename, check_import_digest=False)

    assert any("forbidden_import" in violation for violation in violations), violations


def test_ready_family_routes_factories_recordings_and_plans_are_exactly_equal() -> None:
    module = _family_module()
    registry = module.biorxiv_medrxiv_shadow_registry()
    readiness = module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE)
    family_route_identities = set(_FAMILY_RECORDED_FIXTURES)
    recorded_family_route_ids = {identity[0] for identity in family_route_identities}
    registry_family_route_ids, readiness_family_route_ids, ready_route_ids = _family_boundary_route_ids(
        registry,
        readiness,
    )
    assert registry_family_route_ids == recorded_family_route_ids
    assert readiness_family_route_ids == recorded_family_route_ids
    assert ready_route_ids == recorded_family_route_ids

    ready_routes = tuple(route for route in registry.routes if route.route_id in ready_route_ids)
    registry_route_identities = {(route.route_id, route.adapter_id, route.adapter_version) for route in ready_routes}
    registry_adapter_identities = {(route.adapter_id, route.adapter_version) for route in ready_routes}
    factory = module.biorxiv_medrxiv_gateway_adapters()
    factory_identities = set()
    for adapter_id in factory:
        versions = {route.adapter_version for route in ready_routes if route.adapter_id == adapter_id}
        assert len(versions) == 1
        factory_identities.add((adapter_id, versions.pop()))
    plan_identities = set()
    for route_id in registry_family_route_ids:
        _registry, plan = _family_plan(route_id)
        assert len(plan.dispatch_groups) == 1
        assert plan.dispatch_groups[0].route_id == route_id
        plan_identities.add((plan.dispatch_groups[0].adapter_id, plan.dispatch_groups[0].adapter_version))

    assert registry_route_identities == family_route_identities
    assert registry_adapter_identities == factory_identities == plan_identities
    assert registry_adapter_identities == set(module._FAMILY_PARSING_PROFILES)
    assert set(factory) == {adapter_id for adapter_id, _version in registry_adapter_identities}
    assert {adapter.__module__ for adapter in factory.values()} == {_FAMILY_MODULE_NAME}
    assert all(route.credential_requirement is CredentialRequirement.NONE for route in ready_routes)
    assert all(route.policy.limits.max_redirects == 0 for route in ready_routes)


def test_unrecorded_ready_family_route_cannot_hide_behind_shared_adapter_identity() -> None:
    module = _family_module()
    registry = module.biorxiv_medrxiv_shadow_registry()
    readiness = module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE)
    extra_route_id = "biorxiv_unrecorded_ready_direct"
    extra_route = replace(registry.get_route("biorxiv_details_lookup_direct"), route_id=extra_route_id)
    extra_readiness = replace(
        next(entry for entry in readiness.routes if entry.route_id == "biorxiv_details_lookup_direct"),
        route_id=extra_route_id,
    )
    biorxiv_source = registry.get_source("biorxiv")
    mutated_source = replace(
        biorxiv_source,
        route_references=biorxiv_source.route_references + (SourceRouteReference(extra_route_id, None),),
    )
    mutated_registry = replace(
        registry,
        sources=tuple(mutated_source if source is biorxiv_source else source for source in registry.sources),
        routes=registry.routes + (extra_route,),
    )
    mutated_readiness = replace(readiness, routes=readiness.routes + (extra_readiness,))
    registry_route_ids, readiness_route_ids, ready_route_ids = _family_boundary_route_ids(
        mutated_registry,
        mutated_readiness,
    )
    recorded_route_ids = {identity[0] for identity in _FAMILY_RECORDED_FIXTURES}

    assert registry_route_ids - recorded_route_ids == {extra_route_id}
    assert readiness_route_ids - recorded_route_ids == {extra_route_id}
    assert ready_route_ids - recorded_route_ids == {extra_route_id}


@pytest.mark.asyncio
async def test_all_six_family_routes_use_only_accounted_executor_gateway_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _family_module()
    adapters = module.biorxiv_medrxiv_gateway_adapters(monotonic_clock=lambda: 0.0)
    fixture_bodies = _family_fixture_bodies()
    planned = {identity: _family_plan(identity[0]) for identity in _FAMILY_RECORDED_FIXTURES}
    expected_dispatches = sum(len(bodies) for bodies in fixture_bodies.values())
    dispatch_ids = iter(f"family-dispatch-{index}" for index in range(1, expected_dispatches + 1))
    gateway_calls: list[tuple[str, str]] = []
    executions = {}
    _install_runtime_tripwires(monkeypatch)

    for identity, bodies in fixture_bodies.items():
        route_id, _adapter_id, _adapter_version = identity
        registry, plan = planned[identity]
        remaining = list(bodies)

        async def gateway(
            route,
            intent,
            *,
            is_policy_active,
            expected_route_id=route_id,
            responses=remaining,
        ):
            assert is_policy_active(route.route_id, route.policy.policy_digest)
            assert route.route_id == expected_route_id
            gateway_calls.append((route.route_id, intent.path))
            return _response(route, intent, responses.pop(0))

        execution = await execute_discovery_plan(
            plan,
            registry=registry,
            adapters=adapters,
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: next(dispatch_ids),
            monotonic_clock=lambda: 0.0,
        )
        executions[route_id] = execution
        assert remaining == []
        assert tuple(outcome.state for outcome in execution.logical_outcomes) == (LogicalOutcomeState.SUCCEEDED,)
        assert execution.usage.pages == len(bodies)
        assert len(execution.usage.physical_records) == len(bodies)
        assert execution.usage.accounting.created == len(bodies)
        assert execution.usage.accounting.debited == len(bodies)
        assert execution.candidates
        assert "example.invalid" not in repr(execution)
        assert "fixture-secret" not in repr(execution)

    expected_paths = {
        "biorxiv_europe_pmc_search_aggregator": ["/europepmc/webservices/rest/search"],
        "medrxiv_europe_pmc_search_aggregator": ["/europepmc/webservices/rest/search"],
        "biorxiv_details_lookup_direct": ["/details/biorxiv/10.5555/biorxiv.details.synthetic/na/json"],
        "medrxiv_details_lookup_direct": ["/details/medrxiv/10.5555/medrxiv.details.synthetic/na/json"],
        "biorxiv_details_interval_direct": [
            "/details/biorxiv/2026-06-01/2026-06-02/0/json",
            "/details/biorxiv/2026-06-01/2026-06-02/1/json",
        ],
        "medrxiv_details_interval_direct": [
            "/details/medrxiv/2026-06-01/2026-06-02/0/json",
            "/details/medrxiv/2026-06-01/2026-06-02/1/json",
        ],
    }
    observed_paths = {
        route_id: [path for observed_route_id, path in gateway_calls if observed_route_id == route_id]
        for route_id in expected_paths
    }

    assert observed_paths == expected_paths
    assert len(gateway_calls) == expected_dispatches == 8
    assert Counter(route_id for route_id, _path in gateway_calls) == Counter(
        {
            "biorxiv_europe_pmc_search_aggregator": 1,
            "medrxiv_europe_pmc_search_aggregator": 1,
            "biorxiv_details_lookup_direct": 1,
            "medrxiv_details_lookup_direct": 1,
            "biorxiv_details_interval_direct": 2,
            "medrxiv_details_interval_direct": 2,
        }
    )
    assert executions["biorxiv_details_interval_direct"].usage.accounting.debited == 2
    assert executions["medrxiv_details_interval_direct"].usage.accounting.debited == 2
