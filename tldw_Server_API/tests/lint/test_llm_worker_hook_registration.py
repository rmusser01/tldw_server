"""Every LLM-calling worker registers the Moderation gateway hooks (TASK-13950.27 PR C).

A standalone worker never runs the app lifespan, so without its own call its provider calls reach the
gateway unmoderated. The worker set is the compose ``command:`` modules, the users of
``standalone_llm_provider_overrides*`` and the two named jobs workers. A worker that makes no LLM call
is allowlisted below with its reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT

pytestmark = pytest.mark.unit

COMPOSE = REPO_ROOT / "Dockerfiles" / "docker-compose.workers.yml"
OVERRIDES_DEFINITION = "core/Provider_Credentials/llm_provider_overrides.py"
NAMED = ("core/Explainer/jobs_worker.py", "core/Research/jobs_worker.py")
MARKER = "register_moderation_gateway_hooks"

# worker module path (relative to app/) -> why it makes no LLM call
NO_LLM_CALL = {
    "services/core_jobs_worker.py": "Chatbooks export/import jobs; no provider call",
    "core/File_Artifacts/jobs_worker.py": "renders file artifacts; no provider call",
    "services/privilege_snapshot_worker.py": "builds privilege snapshots; no provider call",
    "core/Collections/reading_digest_jobs_worker.py": "assembles digests from stored items; no provider call",
    "services/connectors_worker.py": "syncs external connector files into ingestion; no provider call",
}


def compose_worker_files(compose: Path, app_root: Path) -> set[str]:
    out = set()
    for module in re.findall(r"command:\s*python -m tldw_Server_API\.app\.([\w.]+)", compose.read_text()):
        out.add(module.replace(".", "/") + ".py")
    return {p for p in out if (app_root / p).exists()}


def overrides_user_files(app_root: Path) -> set[str]:
    return {
        p.relative_to(app_root).as_posix()
        for p in app_root.rglob("*.py")
        if "standalone_llm_provider_overrides" in p.read_text() and p.relative_to(app_root).as_posix() != OVERRIDES_DEFINITION
    }


def unregistered(app_root: Path, compose: Path, allowlist: dict[str, str]) -> list[str]:
    workers = compose_worker_files(compose, app_root) | overrides_user_files(app_root) | {n for n in NAMED if (app_root / n).exists()}
    return sorted(w for w in workers - set(allowlist) if MARKER not in (app_root / w).read_text())


def test_every_llm_worker_registers_the_moderation_hooks():
    assert unregistered(APP_ROOT, COMPOSE, NO_LLM_CALL) == []


def test_the_allowlist_names_only_compose_workers_without_the_marker():
    for rel in NO_LLM_CALL:
        assert rel in compose_worker_files(COMPOSE, APP_ROOT), rel
        assert MARKER not in (APP_ROOT / rel).read_text(), f"{rel} registers; drop it from the allowlist"


def test_the_worker_set_is_not_empty():
    assert len(compose_worker_files(COMPOSE, APP_ROOT)) == 12
    assert overrides_user_files(APP_ROOT)


def test_lint_flags_a_seeded_compose_worker_and_overrides_user(tmp_path):
    app = tmp_path / "app"
    (app / "services").mkdir(parents=True)
    (app / "core").mkdir()
    (app / "services" / "a_worker.py").write_text("def main():\n    pass\n")
    (app / "core" / "b_worker.py").write_text("from x import standalone_llm_provider_overrides\n")
    (app / "core" / "c_worker.py").write_text("from x import standalone_llm_provider_overrides\nregister_moderation_gateway_hooks()\n")
    compose = tmp_path / "c.yml"
    compose.write_text("    command: python -m tldw_Server_API.app.services.a_worker\n")
    assert unregistered(app, compose, {}) == ["core/b_worker.py", "services/a_worker.py"]
    assert unregistered(app, compose, {"services/a_worker.py": "no llm"}) == ["core/b_worker.py"]
