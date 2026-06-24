import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.program_evaluator import ProgramEvaluator


def test_forbidden_imports_detected(monkeypatch):


    pe = ProgramEvaluator()
    code = """
    ```python
    import requests
    print('hello')
    ```
    """
    # Force enabled to test validator path
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    res = pe.evaluate(project_id=None, db=None, llm_output=code, spec={})
    assert res.success is False
    assert "Import not allowed" in (res.error or "")
    assert res.return_code == 3
    assert res.reward == -1.0


def test_runner_python_path_enabled_vs_disabled(tmp_path, monkeypatch):


    # When enabled, reward should reflect metric_var from globals; when disabled, use heuristic
    pe = ProgramEvaluator()
    code = """
    ```python
    val = 2.5
    print('done')
    ```
    """
    spec = {"runner": "python", "objective": "maximize", "metric_var": "val"}
    # Enabled
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    r_on = pe.evaluate(project_id=None, db=None, llm_output=code, spec=spec)
    # reward = 10*(1 - 1/(1+2.5)) ~= 7.142 => score 0.714 in TestRunner
    assert r_on.success is True
    assert r_on.reward > 5.0
    assert r_on.return_code == 0
    assert isinstance(r_on.stdout, str)
    assert isinstance(r_on.stderr, str)
    # Disabled -> Heuristic fallback (no sandbox)
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "false")
    r_off = pe.evaluate(project_id=None, db=None, llm_output=code, spec=spec)
    assert r_off.success is True
    assert r_off.reward <= r_on.reward


def test_code_eval_requires_explicit_unsafe_acknowledgement(monkeypatch):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.delenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", raising=False)

    def _fail_execute(*_args, **_kwargs):
        raise AssertionError("unsafe code execution should stay disabled")

    monkeypatch.setattr(pe, "_execute_in_sandbox", _fail_execute, raising=True)
    res = pe.evaluate(
        project_id=None,
        db=None,
        llm_output="""```python\nval = 9.0\n```""",
        spec={"metric_var": "val", "objective": "maximize"},
    )

    assert res.success is True
    assert res.metrics.get("mode") == "heuristic"
    assert res.metrics.get("code_eval_disabled_reason") == "unsafe_code_eval_not_enabled"


def test_code_eval_requires_production_acknowledgement_in_production(monkeypatch):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("tldw_production", "true")
    monkeypatch.delenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL_IN_PRODUCTION", raising=False)

    def _fail_execute(*_args, **_kwargs):
        raise AssertionError("unsafe production code execution should stay disabled")

    monkeypatch.setattr(pe, "_execute_in_sandbox", _fail_execute, raising=True)
    res = pe.evaluate(
        project_id=None,
        db=None,
        llm_output="""```python\nval = 9.0\n```""",
        spec={"metric_var": "val", "objective": "maximize"},
    )

    assert res.success is True
    assert res.metrics.get("mode") == "heuristic"
    assert res.metrics.get("code_eval_disabled_reason") == "unsafe_code_eval_production_disabled"


def test_code_eval_allows_production_when_explicitly_acknowledged(monkeypatch):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL_IN_PRODUCTION", "true")
    monkeypatch.setenv("APP_ENV", "prod")

    def _fake_execute(code: str, *, timeout_sec: float, memory_mb: int, import_whitelist: set[str]):
        return True, 0, "ok", "", {"val": 2.0}

    monkeypatch.setattr(pe, "_execute_in_sandbox", _fake_execute, raising=True)
    res = pe.evaluate(
        project_id=None,
        db=None,
        llm_output="""```python\nval = 2.0\n```""",
        spec={"metric_var": "val", "objective": "maximize"},
    )

    assert res.success is True
    assert res.metrics.get("mode") == "sandbox"


def test_project_metadata_cannot_enable_code_eval_without_operator_flag(monkeypatch):
    class ProjectMetadataDB:
        def get_project(self, project_id):
            return {"id": project_id, "metadata": {"enable_code_eval": True}}

    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "false")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.delenv("PROMPT_STUDIO_ALLOW_PROJECT_CODE_EVAL", raising=False)

    res = pe.evaluate(
        project_id=1,
        db=ProjectMetadataDB(),
        llm_output="""```python\nval = 9.0\n```""",
        spec={"metric_var": "val", "objective": "maximize"},
    )

    assert res.success is True
    assert res.metrics.get("mode") == "heuristic"
    assert res.metrics.get("code_eval_disabled_reason") == "code_eval_disabled"


def test_program_evaluator_timeout(monkeypatch):


    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_CODE_EVAL_TIMEOUT_MS", "100")
    code = """
    ```python
    x = 0
    while True:
        x += 1
    ```
    """
    res = pe.evaluate(project_id=None, db=None, llm_output=code, spec={})
    assert res.success is False
    assert res.return_code == 124
    assert res.reward == -1.0
    assert isinstance(res.stderr, str)


def test_program_evaluator_import_whitelist_env(monkeypatch):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_CODE_EVAL_IMPORT_WHITELIST", "math")

    allowed_code = """
    ```python
    import math
    val = math.sqrt(9)
    ```
    """
    allowed_res = pe.evaluate(project_id=None, db=None, llm_output=allowed_code, spec={"metric_var": "val"})
    assert allowed_res.success is True
    assert allowed_res.return_code == 0

    blocked_code = """
    ```python
    import statistics
    val = statistics.mean([1,2,3])
    ```
    """
    blocked_res = pe.evaluate(project_id=None, db=None, llm_output=blocked_code, spec={"metric_var": "val"})
    assert blocked_res.success is False
    assert blocked_res.return_code == 3


@pytest.mark.parametrize(
    "snippet",
    [
        "open('/tmp/x', 'w')",
        "import socket\nsocket.socket()",
        "__import__('os').system('echo nope')",
    ],
)
def test_program_evaluator_blocks_unsafe_calls(monkeypatch, snippet):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    code = f"""
    ```python
    {snippet}
    ```
    """
    res = pe.evaluate(project_id=None, db=None, llm_output=code, spec={})
    assert res.success is False
    assert res.return_code == 3
    assert "policy_violation" in (res.metrics or {})


def test_extract_code_handles_markdown_indentation():
    pe = ProgramEvaluator()
    raw = """
    ```python
        x = 0
        while x < 2:
            x += 1
    ```
    """
    code = pe._extract_code(raw)
    assert code is not None
    assert "while x < 2:" in code
    # Ensure nested indentation is preserved after extraction.
    assert "    x += 1" in code


def test_program_evaluator_memory_limit_env_is_applied(monkeypatch):
    pe = ProgramEvaluator()
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_CODE_EVAL_MEM_MB", "64")

    seen = {}

    def _fake_execute(code: str, *, timeout_sec: float, memory_mb: int, import_whitelist: set[str]):
        seen["memory_mb"] = int(memory_mb)
        return True, 0, "ok", "", {"val": 1.0}

    monkeypatch.setattr(pe, "_execute_in_sandbox", _fake_execute, raising=True)
    res = pe.evaluate(
        project_id=None,
        db=None,
        llm_output="""```python\nval = 1.0\n```""",
        spec={"metric_var": "val", "objective": "maximize"},
    )
    assert res.success is True
    assert seen["memory_mb"] == 64
    assert float((res.metrics or {}).get("memory_mb", 0)) == 64.0
