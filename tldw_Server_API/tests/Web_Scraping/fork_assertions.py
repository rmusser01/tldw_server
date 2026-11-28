"""Run genuine fork-inheritance assertions outside pytest's background threads."""

import json
import subprocess
import sys
from pathlib import Path


def run_single_thread_fork_check(module: str, assertion: str, *arguments: object) -> None:
    module = f"tldw_Server_API.tests.Web_Scraping.{module.rsplit('.', 1)[-1]}"
    script = """
import importlib
import json
import sys
import threading
import warnings

module, assertion, arguments = sys.argv[1:]
checks = importlib.import_module(module)
warnings.filterwarnings('error', message='.*multi-threaded.*', category=DeprecationWarning)
assert threading.active_count() == 1, [thread.name for thread in threading.enumerate()]
print('FORK_BOOTSTRAP: MainThread only; native-fork warnings are errors', flush=True)
getattr(checks, assertion)(*json.loads(arguments))
print('FORK_ASSERTION_PASSED: ' + assertion, flush=True)
"""
    # Run the assertion directly, never recursively invoke its pytest wrapper.
    result = subprocess.run(
        [sys.executable, "-c", script, module, assertion, json.dumps(arguments)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"FORK_ASSERTION_PASSED: {assertion}" in result.stdout, result.stdout
