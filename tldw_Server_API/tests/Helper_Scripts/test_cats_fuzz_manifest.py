from __future__ import annotations

import pytest

from Helper_Scripts.cats_fuzz.manifest import (
    BlockRisk,
    ExpectedGate,
    get_builtin_block,
    get_builtin_manifest,
    validate_block,
)


@pytest.mark.unit
def test_builtin_manifest_contains_initial_blocks() -> None:
    manifest = get_builtin_manifest()

    assert {"contract", "public-read", "auth-read"}.issubset(manifest)
    assert manifest["contract"].risk is BlockRisk.CONTRACT
    assert manifest["contract"].expected_gate is ExpectedGate.CONTRACT_ONLY
    assert manifest["contract"].calls_api_service is False
    assert manifest["contract"].paths == ()
    assert manifest["public-read"].expected_gate is ExpectedGate.NO_5XX
    assert manifest["public-read"].blackbox is True
    assert manifest["public-read"].requires_readiness is True
    assert manifest["public-read"].timeout_seconds >= 300
    assert manifest["public-read"].allows_mutation is False
    assert manifest["public-read"].allows_network is False
    assert manifest["public-read"].include_api_key is False
    assert "/" in manifest["public-read"].paths
    assert manifest["auth-read"].allows_mutation is False
    assert manifest["auth-read"].allows_network is False
    assert manifest["auth-read"].include_api_key is True


@pytest.mark.unit
def test_public_read_readiness_paths_require_readiness_metadata() -> None:
    manifest = get_builtin_manifest()
    readiness_paths = {"/ready", "/health/ready", "/api/v1/health/ready"}

    for block in manifest.values():
        covered_readiness_paths = readiness_paths.intersection(block.paths)
        if covered_readiness_paths:
            assert block.requires_readiness is True, block.name


@pytest.mark.unit
def test_mutating_blocks_must_require_seed_or_be_manual() -> None:
    block = get_builtin_block("public-read")
    unsafe = block.__class__(
        **{
            **block.__dict__,
            "name": "unsafe",
            "allows_mutation": True,
            "requires_seed": False,
        }
    )

    with pytest.raises(ValueError, match="requires_seed"):
        validate_block(unsafe)


@pytest.mark.unit
def test_manual_blocks_may_mutate_without_seed() -> None:
    block = get_builtin_block("public-read")
    manual = block.__class__(
        **{
            **block.__dict__,
            "name": "manual-check",
            "risk": BlockRisk.MANUAL,
            "allows_mutation": True,
            "requires_seed": False,
        }
    )

    validate_block(manual)


@pytest.mark.unit
def test_network_enabled_blocks_must_be_external_risk() -> None:
    block = get_builtin_block("public-read")
    unsafe = block.__class__(**{**block.__dict__, "name": "unsafe-network", "allows_network": True})

    with pytest.raises(ValueError, match="external-risk"):
        validate_block(unsafe)


@pytest.mark.unit
def test_unknown_builtin_block_fails() -> None:
    with pytest.raises(KeyError):
        get_builtin_block("missing")
