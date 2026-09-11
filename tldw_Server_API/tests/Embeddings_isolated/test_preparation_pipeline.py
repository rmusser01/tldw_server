from __future__ import annotations

from collections.abc import Callable

import pytest

from tldw_Server_API.app.core.Embeddings import preparation as preparation_module
from tldw_Server_API.app.core.Embeddings.preparation import (
    EmbeddingPreparationPipeline,
    effective_dimension_policy,
)
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingRequestContext


def _context() -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="sensitive-user-marker",
        model_field="sentence-transformers/all-MiniLM-L6-v2",
        provider_header="huggingface",
        dimensions=None,
        encoding_format="float",
        request_id="sensitive-request-marker",
    )


def _tokens_to_texts(
    tokens_input: list[int] | list[list[int]],
    model: str,
) -> tuple[list[str], int, list[int]]:
    del model
    if tokens_input and isinstance(tokens_input[0], int):
        return ["decoded"], len(tokens_input), [len(tokens_input)]
    token_batches = tokens_input
    return [f"decoded-{index}" for index, _ in enumerate(token_batches)], 0, [len(item) for item in token_batches]


def _pipeline(**overrides: object) -> EmbeddingPreparationPipeline:
    kwargs: dict[str, object] = {
        "count_tokens": lambda text, model: len(text.split()),
        "tokens_to_texts": _tokens_to_texts,
        "settings_config": {},
        "max_tokens": 128,
        "implemented_providers": {"huggingface", "openai"},
        "allowed_providers": None,
        "allowed_models": None,
        "enforce_policy": True,
        "allow_fallback_with_header": True,
        "settings_fallback_chain": None,
        "settings_fallback_model_map": None,
        "dimension_policy": "reduce",
        "require_model": True,
        "guess_provider": None,
        "backend_identity_resolver": lambda provider, model: f"{provider}:{model}:backend",
        "cache_namespace": None,
        "batch_size": None,
        "execution_path": "legacy",
    }
    kwargs.update(overrides)
    return EmbeddingPreparationPipeline(**kwargs)


def _install_boundary_probes(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    failing_boundary: str | None = None,
    failure: Exception | None = None,
) -> Callable[[str, str], str]:
    real_resolve = preparation_module.resolve_provider_model
    real_normalize = preparation_module.normalize_embedding_input
    real_policy = preparation_module.enforce_embedding_policy

    def maybe_raise(boundary: str) -> None:
        if boundary == failing_boundary:
            assert failure is not None
            raise failure

    def resolve_probe(*args, **kwargs):
        calls.append("resolve_provider_model")
        maybe_raise("resolve_provider_model")
        return real_resolve(*args, **kwargs)

    def normalize_probe(*args, **kwargs):
        calls.append("normalize_embedding_input")
        maybe_raise("normalize_embedding_input")
        return real_normalize(*args, **kwargs)

    def policy_probe(*args, **kwargs):
        calls.append("enforce_embedding_policy")
        maybe_raise("enforce_embedding_policy")
        return real_policy(*args, **kwargs)

    def identity_probe(provider: str, model: str) -> str:
        calls.append("backend_identity_resolver")
        maybe_raise("backend_identity_resolver")
        return f"{provider}:{model}:backend"

    monkeypatch.setattr(preparation_module, "resolve_provider_model", resolve_probe)
    monkeypatch.setattr(preparation_module, "normalize_embedding_input", normalize_probe)
    monkeypatch.setattr(preparation_module, "enforce_embedding_policy", policy_probe)
    return identity_probe


@pytest.mark.unit
def test_pipeline_reports_only_phase_strings_before_boundaries_in_exact_order(monkeypatch):
    events: list[str] = []
    sink_calls: list[tuple[object, ...]] = []
    identity_probe = _install_boundary_probes(monkeypatch, events)

    def phase_sink(*args: object) -> None:
        sink_calls.append(args)
        events.append(f"phase:{args[0]}")

    prepared = _pipeline(backend_identity_resolver=identity_probe).prepare(
        "sensitive-raw-input-marker",
        _context(),
        phase_sink=phase_sink,
    )

    assert events == [
        "phase:resolving_intent",
        "resolve_provider_model",
        "phase:normalizing",
        "normalize_embedding_input",
        "phase:resolving_policy",
        "enforce_embedding_policy",
        "phase:planning",
        "backend_identity_resolver",
    ]
    assert sink_calls == [
        ("resolving_intent",),
        ("normalizing",),
        ("resolving_policy",),
        ("planning",),
    ]
    assert all(isinstance(args[0], str) for args in sink_calls)
    assert "sensitive-raw-input-marker" not in repr(sink_calls)
    assert "sensitive-user-marker" not in repr(sink_calls)
    assert "sensitive-request-marker" not in repr(sink_calls)
    assert prepared.execution_plan.observability_tags not in sink_calls


@pytest.mark.unit
def test_pipeline_phase_sink_is_optional_and_does_not_change_prepared_output():
    without_sink = _pipeline().prepare("optional sink", _context())
    phases: list[str] = []

    with_sink = _pipeline().prepare("optional sink", _context(), phase_sink=phases.append)

    assert without_sink == with_sink
    assert phases == ["resolving_intent", "normalizing", "resolving_policy", "planning"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failing_boundary", "expected_phases", "expected_calls"),
    [
        ("resolve_provider_model", ["resolving_intent"], ["resolve_provider_model"]),
        (
            "normalize_embedding_input",
            ["resolving_intent", "normalizing"],
            ["resolve_provider_model", "normalize_embedding_input"],
        ),
        (
            "enforce_embedding_policy",
            ["resolving_intent", "normalizing", "resolving_policy"],
            [
                "resolve_provider_model",
                "normalize_embedding_input",
                "enforce_embedding_policy",
            ],
        ),
        (
            "backend_identity_resolver",
            ["resolving_intent", "normalizing", "resolving_policy", "planning"],
            [
                "resolve_provider_model",
                "normalize_embedding_input",
                "enforce_embedding_policy",
                "backend_identity_resolver",
            ],
        ),
    ],
)
def test_pipeline_boundary_failure_preserves_identity_and_stops_later_work(
    monkeypatch,
    failing_boundary,
    expected_phases,
    expected_calls,
):
    sentinel = RuntimeError(f"sentinel:{failing_boundary}")
    calls: list[str] = []
    phases: list[str] = []
    identity_probe = _install_boundary_probes(
        monkeypatch,
        calls,
        failing_boundary=failing_boundary,
        failure=sentinel,
    )

    with pytest.raises(RuntimeError) as exc_info:
        _pipeline(backend_identity_resolver=identity_probe).prepare(
            "boundary failure",
            _context(),
            phase_sink=phases.append,
        )

    assert exc_info.value is sentinel
    assert phases == expected_phases
    assert calls == expected_calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failing_phase", "expected_phases", "expected_calls"),
    [
        ("resolving_intent", ["resolving_intent"], []),
        ("normalizing", ["resolving_intent", "normalizing"], ["resolve_provider_model"]),
        (
            "resolving_policy",
            ["resolving_intent", "normalizing", "resolving_policy"],
            ["resolve_provider_model", "normalize_embedding_input"],
        ),
        (
            "planning",
            ["resolving_intent", "normalizing", "resolving_policy", "planning"],
            [
                "resolve_provider_model",
                "normalize_embedding_input",
                "enforce_embedding_policy",
            ],
        ),
    ],
)
def test_pipeline_sink_failure_preserves_identity_and_skips_associated_boundary(
    monkeypatch,
    failing_phase,
    expected_phases,
    expected_calls,
):
    sentinel = RuntimeError(f"sentinel:{failing_phase}")
    calls: list[str] = []
    phases: list[str] = []
    identity_probe = _install_boundary_probes(monkeypatch, calls)

    def failing_sink(phase: str) -> None:
        phases.append(phase)
        if phase == failing_phase:
            raise sentinel

    with pytest.raises(RuntimeError) as exc_info:
        _pipeline(backend_identity_resolver=identity_probe).prepare(
            "sink failure",
            _context(),
            phase_sink=failing_sink,
        )

    assert exc_info.value is sentinel
    assert phases == expected_phases
    assert calls == expected_calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("encoding_format", "dimensions", "configured_policy", "expected"),
    [
        ("base64", 128, "ignore", "reduce"),
        ("base64", None, "pad", "pad"),
        ("float", 128, "ignore", "ignore"),
        (None, 128, "pad", "pad"),
    ],
)
def test_effective_dimension_policy(
    encoding_format,
    dimensions,
    configured_policy,
    expected,
):
    assert effective_dimension_policy(encoding_format, dimensions, configured_policy) == expected
