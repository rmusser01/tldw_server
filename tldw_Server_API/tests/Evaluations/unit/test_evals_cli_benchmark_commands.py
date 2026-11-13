from click.testing import CliRunner
from loguru import logger

from tldw_Server_API.cli.evals_cli import main


def test_unified_cli_help_includes_benchmark_group():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.output


def test_unified_cli_has_list_benchmarks_alias():
    result = CliRunner().invoke(main, ["list-benchmarks", "--help"])
    assert result.exit_code == 0


def test_benchmark_run_command_executes(monkeypatch):
    called = {"loaded": False}

    class _Cfg:
        evaluation_type = "bullshit_detection"

    class _Registry:
        def get(self, name):
            return _Cfg() if name == "bullshit_benchmark" else None

        def create_evaluator(self, name):
            if name != "bullshit_benchmark":
                return None

            class _Eval:
                def format_for_custom_metric(self, item):
                    return {
                        "name": "m",
                        "description": "d",
                        "evaluation_prompt": "{question}",
                        "input_data": {"question": item["question"]},
                        "scoring_criteria": {"k": "v"},
                    }

            return _Eval()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.benchmark_cli.get_registry",
        lambda: _Registry(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.benchmark_cli.load_benchmark_dataset",
        lambda name, source=None, limit=None, **kwargs: (
            called.__setitem__("loaded", True)
            or [{"id": "q1", "question": "q", "nonsensical_element": "n", "domain": "d"}]
        ),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.cli.benchmark_cli.run_async_safely",
        lambda coro: [{"score": 1.0, "explanation": "ok"}],
    )

    result = CliRunner().invoke(
        main,
        ["benchmark", "run", "bullshit_benchmark", "--limit", "1"],
    )
    assert result.exit_code == 0
    assert called["loaded"] is True


def test_benchmark_run_summary_leaves_an_unscored_item_out(monkeypatch, tmp_path):
    """An unscored item (score None) is counted apart and left out of the average and accuracy, on screen and in
    the saved results (owner, 2026-11-10; TASK-15485)."""
    import json

    class _Registry:
        def get(self, name):
            return type("Cfg", (), {"evaluation_type": "bullshit_detection"})()

        def create_evaluator(self, name):
            return type("Eval", (), {"format_for_custom_metric": lambda self, item: {}})()

    cli = "tldw_Server_API.app.core.Evaluations.cli.benchmark_cli"
    monkeypatch.setattr(f"{cli}.get_registry", lambda: _Registry())
    monkeypatch.setattr(f"{cli}.load_benchmark_dataset", lambda *a, **k: [{"id": "q1"}, {"id": "q2"}])
    # "category" sends the summary through calculate_accuracy_by_category too
    monkeypatch.setattr(
        f"{cli}.run_async_safely",
        lambda coro: coro.close() or [
            {"score": 0.8, "explanation": "ok", "category": "c"},
            {"score": None, "explanation": "Error: 429 rate limited", "category": "c"},
        ],
    )
    out = tmp_path / "results.json"

    result = CliRunner().invoke(main, ["benchmark", "run", "bullshit_benchmark", "--output", str(out)])

    assert result.exit_code == 0, result.output
    for line in ("Unscored: 1", "Accuracy: 100.0%", "Average Score: 0.800", "c: 100.0% (1/1)"):
        assert line in result.output
    assert json.loads(out.read_text())["summary"] == {"total": 2, "unscored": 1, "average_score": 0.8}


def test_cli_log_sink_does_not_error_after_clirunner_teardown(capsys):
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0

    logger.info("post-clirunner-log")
    captured = capsys.readouterr()
    assert "Logging error in Loguru Handler" not in captured.err
