"""Built-in CATS fuzzing block manifest and validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockRisk(str, Enum):
    """Risk categories used to group harness blocks by allowed behavior."""

    CONTRACT = "contract"
    PUBLIC_READ = "public-read"
    AUTH_READ = "auth-read"
    ISOLATED_MUTATION = "isolated-mutation"
    EXTERNAL_RISK = "external-risk"
    MANUAL = "manual"


class ExpectedGate(str, Enum):
    """Expected CI gate semantics for a harness block."""

    NO_5XX = "no_5xx"
    CONTRACT_ONLY = "contract_only"


@dataclass(frozen=True)
class CatsBlock:
    """Declarative configuration for one CATS fuzzing block."""

    name: str
    description: str
    risk: BlockRisk
    paths: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    skip_paths: tuple[str, ...] = ()
    skip_tags: tuple[str, ...] = ()
    skip_methods: tuple[str, ...] = ()
    requires_seed: bool = False
    allows_mutation: bool = False
    allows_network: bool = False
    calls_api_service: bool = True
    include_api_key: bool = True
    blackbox: bool = True
    requires_readiness: bool = False
    timeout_seconds: int = 120
    read_timeout: int = 5
    connection_timeout: int = 5
    write_timeout: int = 5
    max_requests_per_minute: int = 120
    expected_gate: ExpectedGate = ExpectedGate.NO_5XX
    skip_reason: str | None = None
    report_formats: tuple[str, ...] = ("HTML_ONLY", "JUNIT")


def validate_block(block: CatsBlock) -> None:
    """Validate that a CATS block obeys the harness safety constraints."""
    if not block.name:
        raise ValueError("block name is required")
    if not block.paths and not block.tags and block.risk is not BlockRisk.CONTRACT:
        raise ValueError(f"{block.name}: paths or tags are required")
    if block.allows_mutation and not block.requires_seed and block.risk is not BlockRisk.MANUAL:
        raise ValueError(f"{block.name}: mutating blocks must set requires_seed")
    if block.allows_network and block.risk is not BlockRisk.EXTERNAL_RISK:
        raise ValueError(f"{block.name}: allows_network requires external-risk")


def get_builtin_manifest() -> dict[str, CatsBlock]:
    """Return the built-in CATS block manifest keyed by block name."""
    blocks = {
        "contract": CatsBlock(
            name="contract",
            description="Validate and summarize generated OpenAPI without calling the API.",
            risk=BlockRisk.CONTRACT,
            calls_api_service=False,
            blackbox=False,
            expected_gate=ExpectedGate.CONTRACT_ONLY,
            timeout_seconds=60,
        ),
        "public-read": CatsBlock(
            name="public-read",
            description="Fuzz public metadata and health endpoints in blackbox mode.",
            risk=BlockRisk.PUBLIC_READ,
            paths=(
                "/",
                "/health",
                "/ready",
                "/health/ready",
                "/api/v1/health",
                "/api/v1/health/live",
                "/api/v1/health/ready",
                "/api/v1/config/docs-info",
                "/api/v1/config/quickstart",
            ),
            skip_methods=("POST", "PUT", "PATCH", "DELETE", "TRACE"),
            allows_mutation=False,
            allows_network=False,
            include_api_key=False,
            blackbox=True,
            requires_readiness=True,
            expected_gate=ExpectedGate.NO_5XX,
            max_requests_per_minute=60,
            timeout_seconds=300,
        ),
        "auth-read": CatsBlock(
            name="auth-read",
            description="Authenticated read-only smoke fuzzing with X-API-KEY.",
            risk=BlockRisk.AUTH_READ,
            paths=(
                "/api/v1/llm/providers",
                "/api/v1/mcp/status",
                "/api/v1/rag/health/simple",
            ),
            skip_methods=("POST", "PUT", "PATCH", "DELETE", "TRACE"),
            allows_mutation=False,
            allows_network=False,
            blackbox=True,
            expected_gate=ExpectedGate.NO_5XX,
            max_requests_per_minute=60,
            timeout_seconds=180,
        ),
    }
    for block in blocks.values():
        validate_block(block)
    return blocks


def get_builtin_block(name: str) -> CatsBlock:
    """Return one built-in CATS block by name."""
    return get_builtin_manifest()[name]


__all__ = [
    "BlockRisk",
    "CatsBlock",
    "ExpectedGate",
    "get_builtin_block",
    "get_builtin_manifest",
    "validate_block",
]
