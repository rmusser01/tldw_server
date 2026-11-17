"""`eval custom` and a batch `type: custom` spec call evaluate_custom_metric with its real parameters (TASK-15489)."""

import json

import pytest
from click.testing import CliRunner

from tldw_Server_API.app.core.Evaluations import evaluation_manager
from tldw_Server_API.cli.commands import evaluation as eval_commands
from tldw_Server_API.cli.evals_cli import main


@pytest.fixture
def judge(monkeypatch, tmp_path):
    """Stub the provider call (`analyze`) and keep the manager's database in tmp_path; `reply` is the judge's text."""
    state = {"reply": "SCORE: 10\nA vivid story.", "calls": []}

    def _analyze(api_name, input_data, custom_prompt, api_key, **kwargs):
        state["calls"].append({"api_name": api_name, "input_data": input_data, "prompt": custom_prompt,
                               "model": kwargs.get("model_override")})
        return state["reply"]

    monkeypatch.setattr(evaluation_manager, "analyze", _analyze)
    real_init = evaluation_manager.EvaluationManager.__init__
    monkeypatch.setattr(
        evaluation_manager.EvaluationManager, "__init__",
        lambda self, db_path=None, **kw: real_init(self, db_path=tmp_path / "evals.db", **kw),
    )
    return state


def test_eval_custom_scores_the_text_with_the_prompt(judge):
    result = CliRunner().invoke(main, ["eval", "custom", "creativity", "Once upon a time", "--prompt", "Rate creativity"])

    assert result.exit_code == 0, result.output
    assert "1.000" in result.output and "A vivid story." in result.output
    (call,) = judge["calls"]
    assert call["api_name"] == "openai"
    assert json.loads(call["input_data"]) == {"text": "Once upon a time"}
    assert call["prompt"].startswith("Rate creativity")


def test_eval_custom_without_a_prompt_names_the_metric(judge):
    result = CliRunner().invoke(main, ["eval", "custom", "clarity", "Some text", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert "'clarity'" in judge["calls"][0]["prompt"]


@pytest.mark.parametrize("reply", ["Error: provider timed out", "I cannot rate this."])
def test_eval_custom_prints_a_failed_judge_as_unscored_not_zero_or_success(judge, reply):
    judge["reply"] = reply

    result = CliRunner().invoke(main, ["eval", "custom", "creativity", "text"])

    assert result.exit_code == 1, result.output
    assert "unscored" in result.output and "produced no score" in result.output
    assert "0.000" not in result.output and "completed successfully" not in result.output


def test_eval_custom_has_no_save_option():
    """evaluate_custom_metric stores nothing, so the command does not offer to (TASK-15489)."""
    assert "--save" not in CliRunner().invoke(main, ["eval", "custom", "--help"]).output


def test_custom_model_override_reaches_the_judge_in_single_and_batch_runs(judge):
    result = CliRunner().invoke(main, ["eval", "custom", "clarity", "text", "--model", "judge-single"])
    eval_commands._run_single_evaluation(
        {"type": "custom", "metric_name": "clarity", "text": "text", "model": "judge-batch"},
        "openai", None, None, {},
    )
    assert result.exit_code == 0, result.output
    assert [call["model"] for call in judge["calls"]] == ["judge-single", "judge-batch"]


@pytest.mark.parametrize("reply,status,code", [
    ("SCORE: 10", "completed", 0), ("Error: provider unavailable", "unscored", 1),
])
def test_eval_custom_emits_one_json_document(judge, reply, status, code):
    judge["reply"] = reply
    result = CliRunner().invoke(main, ["eval", "custom", "clarity", "text", "--model", "judge-x", "--format", "json"])

    assert result.exit_code == code, result.output
    payload = json.loads(result.stdout)
    assert (payload["status"], payload["model"]) == (status, "judge-x")


def test_actual_batch_command_emits_one_json_document_and_uses_its_default_model(judge, tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([{"type": "custom", "metric_name": "clarity", "text": "text"}]), encoding="utf-8")
    result = CliRunner().invoke(main, ["--quiet", "eval", "batch", str(batch), "--model", "judge-default", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert judge["calls"][0]["model"] == "judge-default"
    assert json.loads(result.stdout)[0]["model"] == "judge-default"


@pytest.mark.parametrize("command", [
    ["label-choice", "Question", "--labels", "A", "--labels", "B", "--expected", "A"],
    ["nli-factcheck", "claim", "evidence", "--labels", "A", "--labels", "B", "--expected", "A"],
])
@pytest.mark.parametrize("content,expected", [
    (None, None), ('{"metadata":"A"}', None), ('{"label":"A"}', "A"),
])
def test_classification_cli_uses_only_the_assistant_label(monkeypatch, command, content, expected):
    response = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    monkeypatch.setattr(eval_commands, "chat_api_call", lambda **_kwargs: response)
    rendered = []
    monkeypatch.setattr(eval_commands, "print_json", lambda result, _title: rendered.append(result))

    result = CliRunner().invoke(main, ["eval", *command, "--format", "json"])
    assert result.exit_code == 0, result.output
    assert rendered[0]["prediction"] == expected


def test_a_batch_custom_spec_runs_and_an_unscored_one_is_counted_apart(judge, monkeypatch):
    spec = {"type": "custom", "metric_name": "creativity", "text": "t", "prompt": "Rate it"}
    scored = eval_commands._run_single_evaluation(spec, "openai", None, None, {})
    judge["reply"] = "Error: boom"
    unscored = eval_commands._run_single_evaluation(spec, "openai", None, None, {})
    rows: list = []
    monkeypatch.setattr(eval_commands, "print_table", lambda data, title=None: rows.extend(data))
    monkeypatch.setattr(eval_commands, "print_info", lambda *_a, **_k: None)

    eval_commands._display_batch_summary([scored, unscored])

    assert scored["metrics"]["creativity"]["score"] == 1.0 and scored["status"] == "completed"
    assert unscored["metrics"]["creativity"]["score"] is None and unscored["status"] == "unscored"
    assert {row["Metric"]: row["Count"] for row in rows if row["Metric"] != "Success Rate"} == {
        "Total Evaluations": 2, "Successful": 1, "Failed": 0, "Unscored": 1,
    }
