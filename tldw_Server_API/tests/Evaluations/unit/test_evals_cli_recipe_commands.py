from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from tldw_Server_API.app.core.Evaluations.recipe_runs_service import (
    RecipeDefinitionNotLaunchableError,
)
from tldw_Server_API.cli.evals_cli import main
from tldw_Server_API.app.core.exceptions import RecipeEnqueueError


class _Manifest:
    def __init__(
        self,
        recipe_id: str,
        recipe_version: str,
        supported_modes: list[str],
        *,
        launchable: bool = True,
    ):
        self.recipe_id = recipe_id
        self.recipe_version = recipe_version
        self.name = recipe_id.replace("_", " ").title()
        self.description = f"{recipe_id} description"
        self.launchable = launchable
        self.supported_modes = supported_modes
        self.tags = ["demo"]

    def model_dump(self, mode: str = "python"):
        del mode
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "name": self.name,
            "description": self.description,
            "launchable": self.launchable,
            "supported_modes": self.supported_modes,
            "tags": self.tags,
        }


class _Record:
    def __init__(self):
        self.run_id = "recipe_run_123"
        self.recipe_id = "summarization_quality"
        self.recipe_version = "1"
        self.status = "pending"
        self.review_state = "not_required"
        self.dataset_snapshot_ref = "sha256:abc"
        self.dataset_content_hash = "sha256:def"
        self.confidence_summary = None
        self.recommendation_slots = {}
        self.child_run_ids = []
        self.created_at = "2026-03-29T00:00:00Z"
        self.updated_at = None
        self.metadata = {"run_config": {"candidate_model_ids": ["openai:gpt-4.1-mini"]}}

    def model_dump(self, mode: str = "python"):
        del mode
        return {
            "run_id": self.run_id,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "status": self.status,
            "review_state": self.review_state,
            "dataset_snapshot_ref": self.dataset_snapshot_ref,
            "dataset_content_hash": self.dataset_content_hash,
            "confidence_summary": self.confidence_summary,
            "recommendation_slots": self.recommendation_slots,
            "child_run_ids": self.child_run_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class _Report:
    def __init__(self):
        self.run = _Record()
        self.confidence_summary = None
        self.recommendation_slots = {
            "best_overall": {
                "candidate_run_id": None,
                "reason_code": "not_available",
                "explanation": "No recommendation yet.",
                "confidence": None,
                "metadata": {},
            }
        }

    def model_dump(self, mode: str = "python"):
        del mode
        return {
            "run": self.run.model_dump(),
            "confidence_summary": self.confidence_summary,
            "recommendation_slots": self.recommendation_slots,
        }


class _Service:
    def list_manifests(self):
        return [_Manifest("summarization_quality", "1", ["labeled"])]

    def validate_dataset(self, recipe_id: str, *, dataset_id=None, dataset=None, run_config=None):
        assert recipe_id == "summarization_quality"
        assert dataset_id is None
        assert dataset == [{"input": "bad"}]
        assert run_config is None
        return {
            "valid": False,
            "errors": ["dataset must include labels"],
            "dataset_mode": "unlabeled",
            "sample_count": 1,
        }

    def create_run(self, recipe_id: str, *, dataset_id=None, dataset=None, run_config=None, force_rerun=False):
        assert recipe_id == "summarization_quality"
        assert dataset_id is None
        assert dataset == [{"input": "hello", "expected": "hi"}]
        assert run_config == {
            "candidate_model_ids": ["openai:gpt-4.1-mini"],
            "comparison_mode": "leaderboard",
            "weights": {"quality": 1.0},
        }
        assert force_rerun is False
        return _Record()

    def get_report(self, run_id: str):
        assert run_id == "recipe_run_123"
        return _Report()


class _RecordingDB:
    def __init__(self):
        self.updated: dict[str, object] | None = None

    def update_recipe_run(self, run_id: str, **kwargs):
        self.updated = {"run_id": run_id, **kwargs}
        return True


class _ServiceWithDb(_Service):
    def __init__(self):
        self.db = _RecordingDB()


def test_unified_cli_help_includes_recipes_group():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "recipes" in result.output


def test_recipes_list_command_uses_public_cli(monkeypatch):
    captured: dict[str, str | None] = {}

    def _service_factory(user_id=None, db_path=None):
        captured["user_id"] = user_id
        captured["db_path"] = db_path
        return _Service()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        _service_factory,
    )

    result = CliRunner().invoke(main, ["--db-path", "/tmp/recipes.db", "recipes", "list"])

    assert result.exit_code == 0
    assert "summarization_quality" in result.output
    assert Path(captured["db_path"]).parts[-2:] == ("tmp", "recipes.db")


def test_recipes_validate_dataset_command_outputs_validation_payload(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: _Service(),
    )

    result = CliRunner().invoke(
        main,
        [
            "recipes",
            "validate-dataset",
            "summarization_quality",
            "--dataset-json",
            '[{"input":"bad"}]',
        ],
    )

    assert result.exit_code == 0
    assert '"valid": false' in result.output.lower()
    assert "dataset must include labels" in result.output


def test_recipes_validate_dataset_command_fails_cleanly_for_non_launchable_stub(
    monkeypatch,
):
    class _NonLaunchableService(_Service):
        def validate_dataset(self, recipe_id: str, *, dataset_id=None, dataset=None, run_config=None):
            del recipe_id, dataset_id, dataset, run_config
            raise RecipeDefinitionNotLaunchableError("stub_recipe")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: _NonLaunchableService(),
    )

    result = CliRunner().invoke(
        main,
        [
            "recipes",
            "validate-dataset",
            "stub_recipe",
            "--dataset-json",
            '[{"input":"bad"}]',
        ],
    )

    assert result.exit_code == 1
    assert "not launchable" in result.output


def test_recipes_run_command_enqueues_pending_run(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: _Service(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced.enqueue_recipe_run",
        lambda record, owner_user_id=None, job_manager=None: "job-123",
    )

    result = CliRunner().invoke(
        main,
        [
            "recipes",
            "run",
            "summarization_quality",
            "--dataset-json",
            '[{"input":"hello","expected":"hi"}]',
            "--run-config-json",
            '{"candidate_model_ids":["openai:gpt-4.1-mini"],"comparison_mode":"leaderboard","weights":{"quality":1.0}}',
        ],
    )

    assert result.exit_code == 0
    assert "recipe_run_123" in result.output
    assert "job-123" in result.output


def test_recipes_run_command_marks_failed_when_enqueue_raises(monkeypatch):
    service = _ServiceWithDb()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: service,
    )

    def _raise_enqueue(record, owner_user_id=None, job_manager=None):
        del record, owner_user_id, job_manager
        raise RecipeEnqueueError()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced.enqueue_recipe_run",
        _raise_enqueue,
    )

    result = CliRunner().invoke(
        main,
        [
            "recipes",
            "run",
            "summarization_quality",
            "--dataset-json",
            '[{"input":"hello","expected":"hi"}]',
            "--run-config-json",
            '{"candidate_model_ids":["openai:gpt-4.1-mini"],"comparison_mode":"leaderboard","weights":{"quality":1.0}}',
        ],
    )

    assert result.exit_code != 0
    assert "Failed to enqueue recipe run" in result.output
    assert service.db.updated is not None
    assert service.db.updated["run_id"] == "recipe_run_123"
    assert getattr(service.db.updated["status"], "value", service.db.updated["status"]) == "failed"
    metadata = service.db.updated["metadata"]
    assert metadata["jobs"]["worker_state"] == "enqueue_failed"
    assert metadata["jobs"]["error"] == "recipe_run_enqueue_failed"
    assert metadata["jobs"]["error_message"] == "Failed to enqueue recipe run."


def test_recipes_run_command_rejects_non_object_run_config_json(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: _Service(),
    )

    result = CliRunner().invoke(
        main,
        [
            "recipes",
            "run",
            "summarization_quality",
            "--dataset-json",
            '[{"input":"hello","expected":"hi"}]',
            "--run-config-json",
            '["not-an-object"]',
        ],
    )

    assert result.exit_code != 0
    assert "--run-config-json must be a JSON object." in result.output


def test_recipes_report_command_outputs_report_payload(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced._get_recipe_runs_service",
        lambda user_id=None, db_path=None: _Service(),
    )

    result = CliRunner().invoke(main, ["recipes", "report", "recipe_run_123"])

    assert result.exit_code == 0
    assert '"run_id": "recipe_run_123"' in result.output
    assert '"best_overall"' in result.output
