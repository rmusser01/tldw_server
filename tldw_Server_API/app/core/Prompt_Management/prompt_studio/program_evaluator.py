"""
program_evaluator.py
Sandboxed Program Evaluator (Phase 2)

Executes extracted Python code in a restricted subprocess with resource limits
and evaluates objective/constraints to produce a reward in [-1..10].

Notes:
- Network and filesystem access are blocked via static checks and isolated mode.
- Per-project feature toggle is supported via project metadata or env var.
"""

import json
import os
import re
# The subprocess evaluator is gated by PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL.
import subprocess  # nosec B404
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

_DEFAULT_IMPORT_WHITELIST = ("math", "statistics")
_WRAPPER_ALLOWED_IMPORTS = {"json", "sys", "builtins", "resource", "math"}
_FORBIDDEN_PATTERNS = [
    r"\bos\.system\(",
    r"\bsubprocess\.",
    r"\bopen\(",
    r"\burllib\.",
    r"\brequests\.",
    r"\bsocket\.",
    r"\b__import__\(",
]


@dataclass
class EvalResult:
    success: bool
    reward: float
    metrics: dict[str, Any]
    return_code: int
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class ProgramEvaluator:
    """Sandboxed evaluator for python code in Prompt Studio (Phase 2).

    Usage:
      - feature toggle via env PROMPT_STUDIO_ENABLE_CODE_EVAL
      - execution requires explicit PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL acknowledgement
      - extract code from LLM output (``` fences preferred)
      - execute under resource limits and isolated mode
      - evaluate objective/constraints from test case spec or stdout JSON
    """

    # Default execution limits
    CPU_TIME_SEC = 4
    WALL_TIME_SEC = 8
    MEMORY_MB = 256
    MAX_OUTPUT_CHARS = 4000
    _TRUE_VALUES = {"1", "true", "yes", "on"}

    @staticmethod
    def _env_truthy(name: str, default: str = "false") -> bool:
        return str(os.getenv(name, default)).strip().lower() in ProgramEvaluator._TRUE_VALUES

    @staticmethod
    def is_enabled_globally() -> bool:
        return ProgramEvaluator._env_truthy("PROMPT_STUDIO_ENABLE_CODE_EVAL")

    @staticmethod
    def is_unsafe_execution_acknowledged() -> bool:
        return ProgramEvaluator._env_truthy("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL")

    @staticmethod
    def is_project_metadata_enablement_allowed() -> bool:
        return ProgramEvaluator._env_truthy("PROMPT_STUDIO_ALLOW_PROJECT_CODE_EVAL")

    @staticmethod
    def _project_metadata_flag(db, project_id: Optional[int]) -> Optional[bool]:
        if project_id is None:
            return None
        try:
            proj = db.get_project(project_id)
            md = proj.get("metadata") if isinstance(proj, dict) else None
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:
                    md = None
            if isinstance(md, dict):
                flag = md.get("enable_code_eval")
                if isinstance(flag, bool):
                    return flag
        except Exception as metadata_error:
            logger.debug("Program evaluator failed to resolve project metadata flag", exc_info=metadata_error)
        return None

    @staticmethod
    def _resolve_enablement(db, project_id: Optional[int]) -> tuple[bool, str]:
        project_flag = ProgramEvaluator._project_metadata_flag(db, project_id)
        globally_enabled = ProgramEvaluator.is_enabled_globally()
        project_metadata_allowed = ProgramEvaluator.is_project_metadata_enablement_allowed()

        if project_flag is False:
            requested = False
        elif project_flag is True:
            requested = globally_enabled or project_metadata_allowed
        else:
            requested = globally_enabled

        if not requested:
            return False, "code_eval_disabled"
        if not ProgramEvaluator.is_unsafe_execution_acknowledged():
            return False, "unsafe_code_eval_not_enabled"
        return True, ""

    @staticmethod
    def is_enabled_for_project(db, project_id: Optional[int]) -> bool:
        enabled, _reason = ProgramEvaluator._resolve_enablement(db, project_id)
        return enabled

    # --------------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------------
    def evaluate_text_output(self, text: str) -> float:
        """Fallback heuristic reward if sandbox execution is disabled.

        Returns a reward in [-1..10].
        """
        if not text:
            return -1.0
        t = text.lower()
        reward = 0.0
        if "def " in t or "class " in t:
            reward += 3.0
        if "import " in t:
            reward += 1.5
        if "if __name__ == '__main__'" in t or "if __name__ == \"__main__\"" in t:
            reward += 1.0
        for lib in ("numpy", "pandas", "cvxpy", "scipy"):
            if f"import {lib}" in t:
                reward += 0.5
        if any(re.search(p, text) for p in _FORBIDDEN_PATTERNS):
            reward -= 2.0
        return float(max(-1.0, min(10.0, reward)))

    def evaluate(self, *, project_id: Optional[int], db, llm_output: str, spec: Optional[dict[str, Any]] = None) -> EvalResult:
        """End-to-end evaluation: extract code, run sandbox, score.

        spec may include:
          - objective: "minimize" | "maximize"
          - metric_var: variable name in code globals to inspect (float)
          - constraints: list of simple expressions using names from code globals
        """
        timeout_sec = self._resolve_timeout_seconds()
        memory_mb = self._resolve_memory_mb()
        import_whitelist = self._resolve_import_whitelist()

        enabled, disabled_reason = self._resolve_enablement(db, project_id)
        if not enabled:
            # Feature disabled → fallback heuristic
            return EvalResult(
                success=True,
                reward=self.evaluate_text_output(llm_output),
                metrics={
                    "mode": "heuristic",
                    "code_eval_disabled_reason": disabled_reason,
                },
                return_code=0,
            )

        code = self._extract_code(llm_output)
        if not code:
            return EvalResult(
                success=False,
                reward=-1.0,
                metrics={
                    "mode": "sandbox",
                    "timeout_sec": timeout_sec,
                    "memory_mb": memory_mb,
                },
                return_code=2,
                error="No code detected in output",
            )

        # Quick static security scan
        violation = self._find_policy_violation(code, import_whitelist)
        if violation:
            return EvalResult(
                success=False,
                reward=-1.0,
                metrics={
                    "mode": "sandbox",
                    "policy_violation": violation,
                    "timeout_sec": timeout_sec,
                    "memory_mb": memory_mb,
                },
                return_code=3,
                error=violation,
            )

        success, return_code, exec_out, exec_err, globals_dump = self._execute_in_sandbox(
            code,
            timeout_sec=timeout_sec,
            memory_mb=memory_mb,
            import_whitelist=import_whitelist,
        )
        out = self._truncate_text(exec_out)
        err = self._truncate_text(exec_err)
        if not success:
            failure_kind = "timeout" if int(return_code) == 124 else "execution_error"
            return EvalResult(
                success=False,
                reward=-1.0,
                metrics={
                    "mode": "sandbox",
                    "failure_kind": failure_kind,
                    "timeout_sec": timeout_sec,
                    "memory_mb": memory_mb,
                },
                return_code=return_code,
                stdout=out,
                stderr=err,
                error="Execution failed",
            )

        reward, metrics = self._score_from_outputs(exec_out, globals_dump, spec or {})
        metrics.setdefault("timeout_sec", timeout_sec)
        metrics.setdefault("memory_mb", memory_mb)
        return EvalResult(
            success=True,
            reward=reward,
            metrics=metrics,
            return_code=0,
            stdout=out,
            stderr=err,
        )

    # --------------------------------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------------------------------
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract python code from fenced blocks or heuristics."""
        if not text:
            return None
        # Prefer python-fenced blocks while preserving indentation in code body.
        python_blocks = re.findall(r"```python[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        generic_blocks = re.findall(r"```[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        blocks = python_blocks or generic_blocks
        if blocks:
            normalized = [textwrap.dedent(block).strip() for block in blocks if str(block).strip()]
            if normalized:
                return max(normalized, key=len)
        # Heuristic: find regions with many code-like lines
        lines = text.splitlines()
        buf: list[str] = []
        for ln in lines:
            if any(tok in ln for tok in ("def ", "import ", "class ", "return ", "for ", "while ")):
                buf.append(ln)
        code = "\n".join(buf).strip()
        return code or None

    def _resolve_timeout_seconds(self) -> float:
        raw = os.getenv("PROMPT_STUDIO_CODE_EVAL_TIMEOUT_MS")
        if raw is None:
            return float(max(0.1, self.WALL_TIME_SEC))
        try:
            ms = int(raw)
            if ms <= 0:
                raise ValueError
            return max(0.1, ms / 1000.0)
        except Exception:
            return float(max(0.1, self.WALL_TIME_SEC))

    def _resolve_memory_mb(self) -> int:
        raw = os.getenv("PROMPT_STUDIO_CODE_EVAL_MEM_MB")
        if raw is None:
            return int(max(16, self.MEMORY_MB))
        try:
            mb = int(raw)
            if mb <= 0:
                raise ValueError
            return max(16, mb)
        except Exception:
            return int(max(16, self.MEMORY_MB))

    def _resolve_import_whitelist(self) -> set[str]:
        raw = os.getenv("PROMPT_STUDIO_CODE_EVAL_IMPORT_WHITELIST")
        if raw is None:
            return set(_DEFAULT_IMPORT_WHITELIST)
        imports = {part.strip() for part in str(raw).split(",") if part.strip()}
        return imports or set(_DEFAULT_IMPORT_WHITELIST)

    def _extract_import_roots(self, code: str) -> set[str]:
        roots: set[str] = set()
        for match in re.finditer(r"^\s*import\s+([^\n#]+)", code, re.MULTILINE):
            names = match.group(1).split(",")
            for name in names:
                base = name.strip().split(" as ")[0].split(".")[0].strip()
                if base:
                    roots.add(base)
        for match in re.finditer(r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+", code, re.MULTILINE):
            base = match.group(1).split(".")[0].strip()
            if base:
                roots.add(base)
        return roots

    def _find_policy_violation(self, code: str, import_whitelist: set[str]) -> Optional[str]:
        for pat in _FORBIDDEN_PATTERNS:
            if re.search(pat, code):
                return "Forbidden calls detected"
        seen = self._extract_import_roots(code)
        for mod in sorted(seen):
            if mod not in import_whitelist:
                return f"Import not allowed: {mod}"
        return None

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.MAX_OUTPUT_CHARS:
            return text
        return text[: self.MAX_OUTPUT_CHARS] + "...(truncated)"

    def _execute_in_sandbox(
        self,
        code: str,
        *,
        timeout_sec: float,
        memory_mb: int,
        import_whitelist: set[str],
    ) -> tuple[bool, int, str, str, dict[str, Any]]:
        """Execute code in a restricted subprocess using isolated mode.

        Returns (success, return_code, stdout, stderr, globals_dump)
        """
        cpu_lim = max(1, int(self.CPU_TIME_SEC))
        mem_lim = int(memory_mb) * 1024 * 1024
        allowed_imports = set(import_whitelist) | _WRAPPER_ALLOWED_IMPORTS
        allowed_json = json.dumps(sorted(allowed_imports))
        code_json = json.dumps(code)
        wrapper_lines = [
            "import sys, json, builtins, textwrap",
            "# Try to set resource limits (best-effort)",
            "try:",
            "    import resource",
            f"    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_lim}, {cpu_lim}))",
            f"    resource.setrlimit(resource.RLIMIT_AS, ({mem_lim}, {mem_lim}))",
            "except Exception:",
            "    pass",
            "",
            "# Best-effort isolation: disable file I/O via builtins.open",
            "def _blocked(*a, **k):",
            "    raise RuntimeError(\"file/network operations are disabled\")",
            "for name in (\"open\",):",
            "    setattr(builtins, name, _blocked)",
            "",
            "# Guarded import to allow only whitelisted modules",
            f"_allowed = {allowed_json}",
            "_orig_import = builtins.__import__",
            "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):",
            "    base = name.split('.')[0]",
            "    if base not in _allowed:",
            "        raise ImportError(f\"Import not allowed: {name}\")",
            "    return _orig_import(name, globals, locals, fromlist, level)",
            "builtins.__import__ = _guarded_import",
            "",
            "# Execute user code",
            "__code_globals = {}",
            "__code_locals = None",
            # Embed the pre-JSON-encoded string as a safe Python literal
            # Using !r ensures backslashes (e.g., \n) remain escaped for json.loads
            f"__user_code = json.loads({code_json!r})",
            "try:",
            "    # First try to execute as-is",
            "    exec(compile(__user_code, '<sandbox>', 'exec'), __code_globals, __code_locals)",
            "except IndentationError:",
            "    # Fallback: dedent markdown-indented snippets and retry",
            "    __user_code_fixed = textwrap.dedent(__user_code)",
            "    try:",
            "        exec(compile(__user_code_fixed, '<sandbox>', 'exec'), __code_globals, __code_locals)",
            "    except Exception as e2:",
            "        print(f\"__EXEC_ERROR__: {e2}\", file=sys.stderr)",
            "        sys.exit(7)",
            "except Exception as e:",
            "    print(f\"__EXEC_ERROR__: {e}\", file=sys.stderr)",
            "    sys.exit(7)",
            "",
            "# Serialize selected globals back to parent via stdout trailer",
            "def _jsonable(v):",
            "    import math",
            "    if isinstance(v, (int, float, str, bool)) or v is None:",
            "        return True",
            "    if isinstance(v, (list, tuple)):",
            "        return all(_jsonable(x) for x in v[:50])",
            "    if isinstance(v, dict):",
            "        return all(isinstance(k, str) and _jsonable(v[k]) for k in list(v.keys())[:50])",
            "    return False",
            "_dump = {}",
            "for k, v in list(__code_globals.items()):",
            "    if k.startswith('__'):",
            "        continue",
            "    if _jsonable(v):",
            "        _dump[k] = v",
            "print('\\n__GLOBALS_JSON__\\n' + json.dumps(_dump, separators=(',',':')))",
        ]
        wrapper = "\n".join(wrapper_lines)
        # Create temp dir and files
        with tempfile.TemporaryDirectory() as td:
            script_path = os.path.join(td, "sandbox_runner.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapper)
            # Run isolated Python: -I ignores env vars/user site; -B no pyc; no cwd files
            env = {"PYTHONHASHSEED": "0"}
            try:
                # The code under test is untrusted, but this path is explicitly unsafe opt-in.
                proc = subprocess.run(  # nosec B603
                    [sys.executable, "-I", "-B", script_path],
                    cwd=td,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as te:
                return False, 124, te.stdout or "", te.stderr or "timeout", {}
            out, err = proc.stdout or "", proc.stderr or ""
            if proc.returncode != 0:
                return False, int(proc.returncode), out, err, {}
            # Extract globals JSON trailer
            gjson = {}
            if "__GLOBALS_JSON__" in out:
                try:
                    tail = out.split("__GLOBALS_JSON__", 1)[1]
                    gjson = json.loads(tail.strip())
                except Exception:
                    gjson = {}
            return True, 0, out, err, gjson

    def _score_from_outputs(self, stdout: str, globals_dump: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Map execution results to reward in [-1..10]."""
        metrics: dict[str, Any] = {"mode": "sandbox", "constraints_ok": None}

        # If spec provides metric_var and objective, use it
        metric_var = (spec or {}).get("metric_var")
        objective = str((spec or {}).get("objective", "maximize")).lower()
        constraints = (spec or {}).get("constraints") or []

        score_val: Optional[float] = None
        if metric_var and isinstance(globals_dump, dict) and metric_var in globals_dump:
            try:
                score_val = float(globals_dump[metric_var])
                metrics["metric_var_value"] = score_val
            except Exception:
                score_val = None

        # If not found, try to parse last float from stdout
        if score_val is None and stdout:
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", stdout)
            if nums:
                try:
                    score_val = float(nums[-1])
                    metrics["parsed_metric_value"] = score_val
                except Exception as metric_parse_error:
                    logger.debug("Program evaluator failed to parse metric value from objective output", exc_info=metric_parse_error)

        # Evaluate simple constraints: expect expressions like "x >= 0" over globals_dump
        constraints_ok = True
        if constraints and isinstance(globals_dump, dict):
            for expr in constraints:
                if not self._safe_eval_constraint(expr, globals_dump):
                    constraints_ok = False
                    break
        metrics["constraints_ok"] = constraints_ok

        # Map to reward: if no metric was found, give small heuristic
        if score_val is None:
            base = 3.0 if "success" in (stdout or "").lower() else 1.0
            return float(max(-1.0, min(10.0, base - (0 if constraints_ok else 1.5)))), metrics

        # Normalize: if objective is minimize, invert
        # We map into [0..10] based on relative magnitude; absent a scale, use a soft mapping
        val = float(score_val)
        if objective.startswith("min"):
            # Smaller is better; transform to higher reward when closer to 0
            reward = 10.0 / (1.0 + max(0.0, val))
        else:
            # Larger is better; saturating growth
            reward = 10.0 * (1.0 - (1.0 / (1.0 + max(0.0, val))))
        if not constraints_ok:
            reward *= 0.5
        return float(max(-1.0, min(10.0, reward))), metrics

    # --------------------------------------------------------------------------------------------
    # Safe constraint evaluation via AST
    def _safe_eval_constraint(self, expr: str, names: dict[str, Any]) -> bool:
        import ast
        try:
            tree = ast.parse(str(expr), mode="eval")
        except Exception:
            return False

        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
            ast.Name, ast.Load, ast.Constant, ast.Num,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.USub, ast.UAdd,
        )

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
        try:
            env = {k: names.get(k) for k in names}
            return bool(eval(compile(tree, "<constraint>", "eval"), {"__builtins__": {}}, env))  # nosec B307
        except Exception:
            return False
