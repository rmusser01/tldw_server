"""API compatibility and import isolation for core-owned Notes graph models."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.schemas import notes_graph as api_models

pytestmark = pytest.mark.unit

CORE_MODULE = "tldw_Server_API.app.core.Notes_Graph.graph_models"
COMPATIBILITY_SYMBOLS = (
    "Any",
    "BaseModel",
    "ConfigDict",
    "DEFAULT_SEMANTIC_INDEX_SETTINGS",
    "EdgeType",
    "Enum",
    "Field",
    "GraphEdge",
    "GraphFormat",
    "GraphLimits",
    "GraphNode",
    "LEGACY_EDGE_TYPES",
    "Literal",
    "NoteGraphRequest",
    "NoteGraphResponse",
    "NoteLinkCreate",
    "NoteLinkRestore",
    "NoteLinkUpdate",
    "SemanticConversionContext",
    "SemanticEdgeEvidence",
    "SemanticExcerpt",
    "SemanticExcerptPair",
    "SemanticGraphStatus",
    "SemanticTruncationReason",
    "TimeRange",
    "annotations",
    "datetime",
    "field_validator",
    "model_serializer",
    "model_validator",
    "_StrictSemanticModel",
    "_canonical_edge_types",
    "_normalize_optional_nonblank",
    "_SEMANTIC_MAX_TOP_K",
    "_SEMANTIC_MAX_EVIDENCE_PAIRS",
    "_SEMANTIC_MAX_EXCERPT_CODE_POINTS",
    "_SEMANTIC_MAX_EDGE_EVIDENCE_CODE_POINTS",
    "_SEMANTIC_MAX_RESPONSE_EVIDENCE_BYTES",
)


def test_api_compatibility_exports_preserve_exact_core_symbol_identity() -> None:
    core_models = importlib.import_module(CORE_MODULE)
    public_symbols = {name for name in COMPATIBILITY_SYMBOLS if not name.startswith("_")}

    for name in COMPATIBILITY_SYMBOLS:
        assert getattr(api_models, name) is getattr(core_models, name), name
    assert {name for name in vars(api_models) if not name.startswith("_")} == public_symbols
    assert set(api_models.__all__) == public_symbols


def test_core_model_import_does_not_load_the_api_package() -> None:
    script = (
        "import importlib, json, sys\n"
        f"importlib.import_module({CORE_MODULE!r})\n"
        "print(json.dumps([name for name in sys.modules "
        "if name == 'tldw_Server_API.app.api' "
        "or name.startswith('tldw_Server_API.app.api.')]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, "Core model import failed (diagnostics withheld)"
    assert json.loads(result.stdout) == []


def test_api_and_core_models_produce_identical_openapi() -> None:
    core_models = importlib.import_module(CORE_MODULE)

    def openapi(models):
        app = FastAPI()

        def graph(request):
            return request

        graph.__annotations__ = {
            "request": models.NoteGraphRequest,
            "return": models.NoteGraphResponse,
        }
        app.post("/graph")(graph)
        return app.openapi()

    schema = openapi(api_models)
    assert schema == openapi(core_models)
    operation = schema["paths"]["/graph"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/NoteGraphRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/NoteGraphResponse"
    }
