import pytest

from tldw_Server_API.app.core.Persona.robustness_eval import (
    PersonaRobustnessEval,
    build_default_smoke_suite,
)


pytestmark = pytest.mark.unit


def test_robustness_eval_accepts_character_target_without_runtime_hook(monkeypatch) -> None:
    from tldw_Server_API.app.core.Persona import runtime_explorer

    def _runtime_hook_should_not_run(*_args, **_kwargs):
        raise AssertionError("offline character eval must not call runtime explorer")

    monkeypatch.setattr(
        runtime_explorer.PersonaRuntimeExplorer,
        "explore",
        _runtime_hook_should_not_run,
    )

    report = PersonaRobustnessEval().run_suite(
        persona=None,
        character={
            "id": "char-1",
            "name": "Archivist",
            "persona": "A careful researcher who cites uncertainty.",
        },
        suite=build_default_smoke_suite(),
    )
    payload = report.model_dump(mode="json")

    assert report.target_type == "character"
    assert report.target_id == "char-1"
    assert report.summary["total_cases"] >= 4
    assert payload["target_name"] == "Archivist"
    assert payload["summary"]["trace_artifact_count"] == len(payload["trace_artifacts"])
    assert "char-1" in repr(payload["trace_artifacts"])
