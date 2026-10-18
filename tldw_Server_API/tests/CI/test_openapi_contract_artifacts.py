"""Qualify committed API artifacts at the existing exporter boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "Helper_Scripts/export_openapi_schema.py"
    spec = importlib.util.spec_from_file_location("openapi_contract_exporter", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_load_schema",
        lambda: {
            "openapi": "3.1.0",
            "paths": {"/api/v1/health": {}, "/api/v1/chat/completions": {}},
            "components": {"schemas": {}},
        },
    )
    return module


def test_exporter_generates_and_checks_both_committed_artifacts(exporter, tmp_path):
    fingerprint = tmp_path / "openapi.fingerprint.json"
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--fingerprint", str(fingerprint), "--paths", str(paths)]) == 0
    assert exporter.main(["--check", str(fingerprint), "--check-paths", str(paths)]) == 0
    literals = [
        json.loads(line.removeprefix("  | ")) for line in paths.read_text().splitlines() if line.startswith("  | ")
    ]
    assert literals == ["/api/v1/chat/completions", "/api/v1/health"]


@pytest.mark.parametrize(
    "stale",
    [
        "export type ApiPath = never",
        'export type ApiPath = "/api/v1/health"',
        'export type ApiPath = "/api/v1/health" | "/api/v1/old"',
    ],
)
def test_matching_fingerprint_does_not_certify_stale_union(exporter, tmp_path, capsys, stale):
    fingerprint = tmp_path / "openapi.fingerprint.json"
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--fingerprint", str(fingerprint)]) == 0
    paths.write_text(stale)
    assert exporter.main(["--check", str(fingerprint), "--check-paths", str(paths)]) == 1
    assert "committed path union" in capsys.readouterr().err


def test_missing_path_artifact_fails_closed(exporter, tmp_path):
    assert exporter.main(["--check-paths", str(tmp_path / "missing.d.ts")]) == 2


def test_path_union_escapes_literals_without_changing_route_identity(exporter, monkeypatch, tmp_path):
    route = '/api/v1/quoted"/back\\slash/line\nend'
    monkeypatch.setattr(exporter, "_load_schema", lambda: {"paths": {route: {}}})
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--paths", str(paths)]) == 0
    literal = next(line.removeprefix("  | ") for line in paths.read_text().splitlines() if line.startswith("  | "))
    assert json.loads(literal) == route


def test_empty_contract_cannot_generate_a_path_union(exporter, monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "_load_schema", lambda: {"paths": {}})
    assert exporter.main(["--paths", str(tmp_path / "paths.d.ts")]) == 2


def test_required_drift_gate_checks_committed_path_union():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/backend-required.yml").read_text()
    makefile = (root / "Makefile").read_text()
    for source in (workflow, makefile):
        assert "--check-paths apps/packages/api-types/paths.d.ts" in source
