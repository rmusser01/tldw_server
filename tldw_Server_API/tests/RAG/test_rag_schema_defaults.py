"""RAG request models apply configured contextual defaults at call time (TASK-13950.21, D3)."""

import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.RAG import rag_schemas_unified
from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedBatchRequest, UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import resolve_rag_request

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def rag_env(monkeypatch):
    for key in ("RAG_INCLUDE_PARENT_DOCUMENT", "RAG_PARENT_MAX_TOKENS", "RAG_INCLUDE_SIBLING_CHUNKS", "RAG_SIBLING_WINDOW"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rag_schemas_unified, "_CONTEXTUAL_WARNED", set(), raising=False)
    return monkeypatch


def _loader(**rag: str):
    parser = ConfigParser()
    if rag:
        parser.add_section("RAG")
        for key, value in rag.items():
            parser.set("RAG", key, value)

    def _load() -> ConfigParser:
        return parser

    _load.cache_clear = lambda: None
    return _load


@pytest.mark.parametrize(("env", "cfg", "expected"), [
    ({}, {}, (False, 1200, False, 1)),
    ({"RAG_SIBLING_WINDOW": "3", "RAG_INCLUDE_PARENT_DOCUMENT": "TRUE"}, {}, (True, 1200, False, 3)),
    ({"RAG_INCLUDE_PARENT_DOCUMENT": "1"}, {"include_parent_document": "true"}, (False, 1200, False, 1)),  # env set wins; only "true" is true
    ({"RAG_PARENT_MAX_TOKENS": ""}, {"parent_max_tokens": "800", "include_sibling_chunks": "true"}, (False, 800, True, 1)),
    ({"RAG_SIBLING_WINDOW": "3"}, {"sibling_window": "5"}, (False, 1200, False, 3)),  # an env int beats a configured int
    ({"RAG_INCLUDE_PARENT_DOCUMENT": ""}, {"include_parent_document": "true"}, (False, 1200, False, 1)),  # a blank boolean env is set, and false
])
def test_contextual_defaults_keep_the_lazy_settings_rule(env, cfg, expected) -> None:
    from tldw_Server_API.app.core.config_sections.rag import contextual_defaults

    d = contextual_defaults(_loader(**cfg)(), env=env)
    assert (d.include_parent_document, d.parent_max_tokens, d.include_sibling_chunks, d.sibling_window) == expected


@pytest.mark.parametrize(("env", "cfg", "expected"), [
    ({"RAG_DEFAULT_FTS_LEVEL": "CHUNK"}, {}, "chunk"),
    ({"RAG_DEFAULT_FTS_LEVEL": ""}, {"default_fts_level": "chunk"}, ""),
    ({"RAG_DEFAULT_FTS_LEVEL": " chunk"}, {}, " chunk"),
    ({}, {"default_fts_level": "Chunk"}, "chunk"),
    ({}, {}, "media"),
])
def test_default_fts_level_keeps_the_lazy_settings_rule(env, cfg, expected) -> None:
    from tldw_Server_API.app.core.config_sections.rag import default_fts_level

    assert default_fts_level(_loader(**cfg)(), env=env) == expected


def test_configured_value_applies_when_the_request_leaves_the_field_unset(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4", include_parent_document="true"))
    request = UnifiedRAGRequest(query="q")
    assert (request.sibling_window, request.include_parent_document) == (4, True)
    assert "sibling_window" not in request.model_fields_set
    assert UnifiedBatchRequest(queries=["q"]).sibling_window == 4


def test_an_explicit_request_value_wins(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    rag_env.setenv("RAG_INCLUDE_PARENT_DOCUMENT", "true")
    request = UnifiedRAGRequest(query="q", sibling_window=0, include_parent_document=False)
    assert (request.sibling_window, request.include_parent_document) == (0, False)


@pytest.mark.parametrize("model", ["single", "batch"])
def test_an_explicit_value_equal_to_the_default_stays_set(rag_env, model) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    request = (UnifiedRAGRequest(query="q", sibling_window=1) if model == "single"
               else UnifiedBatchRequest(queries=["q"], sibling_window=1))
    assert request.sibling_window == 1 and "sibling_window" in request.model_fields_set


def test_resolve_rag_request_carries_the_configured_value(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    resolved = resolve_rag_request(UnifiedRAGRequest(query="q"), single_user_id_resolver=lambda: 1)
    assert resolved.payload["sibling_window"] == 4


def test_a_profile_beats_the_configured_value(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4", include_parent_document="true"))
    resolved = resolve_rag_request(
        UnifiedRAGRequest(query="q", rag_profile="fast"),
        get_profile_kwargs_fn=lambda name: {"sibling_window": 2, "include_parent_document": False},
        single_user_id_resolver=lambda: 1,
    )
    assert (resolved.payload["sibling_window"], resolved.payload["include_parent_document"]) == (2, False)


def test_unset_config_gives_todays_literals(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader())
    request = UnifiedRAGRequest(query="q")
    assert (request.include_sibling_chunks, request.sibling_window, request.include_parent_document, request.parent_max_tokens) == (False, 1, False, 1200)


@pytest.mark.parametrize("broken", ["load_error", "out_of_range", "bad_value"])
def test_config_errors_fall_back_to_the_literals_and_log(rag_env, broken) -> None:
    if broken == "load_error":
        def _raise():
            raise OSError("config.txt unreadable")

        _raise.cache_clear = lambda: None
        rag_env.setattr(config, "load_comprehensive_config", _raise)
    elif broken == "out_of_range":
        rag_env.setattr(config, "load_comprehensive_config", _loader())
        rag_env.setenv("RAG_SIBLING_WINDOW", "100")  # above the field's le=50
    else:
        rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
        rag_env.setenv("RAG_PARENT_MAX_TOKENS", "abc")  # one bad key: all four keep their literals
    seen: list[str] = []
    sink = logger.add(seen.append, level="WARNING")
    try:
        request = UnifiedRAGRequest(query="q")
    finally:
        logger.remove(sink)
    assert request.sibling_window == 1
    assert any("RAG contextual default" in message for message in seen)


def test_importing_rag_schemas_does_not_load_lazy_settings() -> None:
    # The RAG package __init__ pulls in the pipeline, which loads LazySettings through
    # unrelated modules; stub the package so only the schema module's own imports count.
    code = (
        "import sys, types; "
        "import tldw_Server_API.app.core as core; "
        "pkg = types.ModuleType('tldw_Server_API.app.core.RAG'); "
        "pkg.__path__ = [core.__path__[0] + '/RAG']; "
        "sys.modules['tldw_Server_API.app.core.RAG'] = pkg; "
        "import tldw_Server_API.app.core.RAG.rag_schemas_unified; "
        "from tldw_Server_API.app.core import config; "
        "print(config.settings._data is None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "True"
