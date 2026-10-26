"""Ratchet: every *TEST* env read in app/ is guarded or allowlisted with a reason (TASK-13950.6).

"Guarded" means named by core/testing.GUARDED_TEST_SEAM_KEYS or GUARDED_TEST_SEAM_PREFIXES:
validate_test_runtime_flags() refuses to start a production-like runtime with one armed,
PYTEST_CURRENT_TEST or not. Any other name containing TEST must be listed in ALLOWLIST
with a reason and the exact modules that read it, so a new seam, or a new reader of an
unguarded one such as PYTEST_CURRENT_TEST, is a reviewed change.

Counted: every string constant in app/ shaped like an env name (``[A-Z0-9_]*TEST[A-Z0-9_]*``).
AST-derived, so getenv, environ.get, environ[...], env_flag_enabled("...") and key
tuples all count, and comments and docstrings never do. Test modules shipped inside
app/ (``app/**/tests/``) are skipped. Being greedy, it also matches domain words
(CONTESTED, ABTEST); those are allowlisted as not-a-seam. An allowlisted
(name, module) pair that is no longer found is stale and fails.

Counter: ``python tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py``
prints the unguarded names and their modules. Seed: 38 names in 87 (name, module) pairs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from tldw_Server_API.app.core.testing import GUARDED_TEST_SEAM_KEYS, GUARDED_TEST_SEAM_PREFIXES

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-required.yml"
THIS_FILE = "tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py"
GUARD_FILE = "tldw_Server_API/app/core/testing.py"
# The guard and refusal tests this task adds; the required gate must keep running them.
GUARD_TESTS = (
    "tldw_Server_API/tests/Services/test_startup_test_runtime_guard.py",
    "tldw_Server_API/tests/AuthNZ/unit/test_runtime_environment.py",
    "tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py",
)
_TEST_NAME = re.compile(r"[A-Z0-9_]*TEST[A-Z0-9_]*")
# TEST as its own word (TEST_MODE, IN_TESTS, PYTEST_X); otherwise it is inside a word (LATEST).
_SEAM_SHAPED = re.compile(r"(?:^|_)(?:PY)?TEST(?:S|ING)?(?:_|$)")

_UNDER_TEST_MODE = (
    "only consulted while TEST_MODE/TESTING is on, which startup refuses in a production-like runtime"
)
_UNDER_PYTEST = "only consulted under PYTEST_CURRENT_TEST, and only to opt back into production behaviour"
_NOT_A_SEAM = "not a test seam: TEST is part of a domain word (A/B test, contested claim, test cases, probe)"
_PYTEST_MARKER = (
    "pytest's own marker, set on every pytest run, so startup cannot refuse it outright. In a "
    "production-like runtime startup refuses it unless pytest is actually loaded, and the "
    "AuthNZ JWT/API-key fallbacks ignore it. This list covers modules that name the literal; "
    "callers of core/testing.is_explicit_pytest_runtime() are not listed"
)

# env name -> (reason, modules under app/ that read it). Adding a name or a module is a
# baseline bump: give the reason here and get it reviewed in the PR. Remove pairs as the
# reads go away (the stale check enforces it).
ALLOWLIST: dict[str, tuple[str, tuple[str, ...]]] = {
    "PYTEST_CURRENT_TEST": (
        _PYTEST_MARKER,
        (
            "__init__.py",
            "api/v1/API_Deps/Audit_DB_Deps.py",
            "api/v1/endpoints/admin/__init__.py",
            "api/v1/endpoints/character_chat_sessions.py",
            "api/v1/endpoints/embeddings_v5_production_enhanced.py",
            "api/v1/endpoints/mcp_unified_endpoint.py",
            "api/v1/endpoints/prompt_studio/prompt_studio_evaluations.py",
            "api/v1/endpoints/writing.py",
            "core/Audit/unified_audit_service.py",
            "core/AuthNZ/auth_principal_resolver.py",
            "core/AuthNZ/byok_testing.py",
            "core/AuthNZ/database.py",
            "core/AuthNZ/key_resolution.py",
            "core/AuthNZ/llm_budget_guard.py",
            "core/AuthNZ/llm_budget_middleware.py",
            "core/AuthNZ/migrations.py",
            "core/AuthNZ/session_manager.py",
            "core/AuthNZ/virtual_keys.py",
            "core/Chat/chat_service.py",
            "core/Chat/conversation_enrichment.py",
            "core/Chatbooks/quota_manager.py",
            "core/Chunking/chunker.py",
            "core/DB_Management/db_path_utils.py",
            "core/DB_Management/media_db/runtime/backend_resolution.py",
            "core/Embeddings/audit_adapter.py",
            "core/Evaluations/audit_adapter.py",
            "core/Evaluations/user_rate_limiter.py",
            "core/Ingestion_Media_Processing/Audio/Audio_Files.py",
            "core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py",
            "core/Ingestion_Media_Processing/download_utils.py",
            "core/MCP_unified/environment.py",
            "core/Prompt_Management/prompt_studio/evaluation_manager.py",
            "core/RAG/rag_custom_metrics.py",
            "core/Resource_Governance/governor_redis.py",
            "core/Sandbox/runners/lima_enforcer.py",
            "core/Scheduler/config.py",
            "core/Security/egress.py",
            "core/TTS/adapters/chatterbox_adapter.py",
            "core/TTS/adapters/dia_adapter.py",
            "core/TTS/adapters/higgs_adapter.py",
            "core/TTS/adapters/vibevoice_adapter.py",
            "core/TTS/voice_manager.py",
            "core/testing.py",
            "main.py",
        ),
    ),
    "ALLOW_TEST_MODE_IN_PRODUCTION": (_UNDER_TEST_MODE, ("api/v1/API_Deps/auth_deps.py",)),
    "FORCE_CHAT_QUEUE_IN_TESTS": (_UNDER_TEST_MODE, ("api/v1/endpoints/chat.py",)),
    "HEALTH_EXPOSE_TEST_API_KEY": (_UNDER_TEST_MODE, ("api/v1/endpoints/health.py",)),
    "JOBS_SSE_TEST_MAX_SECONDS": (
        _UNDER_TEST_MODE,
        ("api/v1/endpoints/jobs_admin.py", "api/v1/endpoints/media/ingest_jobs.py", "core/Jobs/settings.py"),
    ),
    "RESEARCH_RUNS_SSE_TEST_MAX_SECONDS": (_UNDER_TEST_MODE, ("api/v1/endpoints/research_runs.py",)),
    "TEST_CHAT_BURST_MULTIPLIER": (_UNDER_TEST_MODE, ("core/Chat/rate_limiter.py",)),
    "TEST_CHAT_GLOBAL_RPM": (_UNDER_TEST_MODE, ("core/Chat/rate_limiter.py",)),
    "TEST_CHAT_PER_CONVERSATION_RPM": (_UNDER_TEST_MODE, ("core/Chat/rate_limiter.py",)),
    "TEST_CHAT_PER_USER_RPM": (_UNDER_TEST_MODE, ("core/Chat/rate_limiter.py",)),
    "TEST_CHAT_TOKENS_PER_MINUTE": (_UNDER_TEST_MODE, ("core/Chat/rate_limiter.py",)),
    "USE_REAL_OPENAI_IN_TESTS": (_UNDER_TEST_MODE, ("api/v1/endpoints/embeddings_v5_production_enhanced.py",)),
    "WORKFLOWS_TEST_REPLAY_SUCCESS": (_UNDER_TEST_MODE, ("api/v1/endpoints/workflows.py",)),
    "CHAT_ALLOW_OPENROUTER_DISCOVERY_IN_TESTS": (_UNDER_PYTEST, ("core/Chat/chat_service.py",)),
    "MINIMAL_TEST_INCLUDE_AUDIO": (_UNDER_PYTEST, ("core/testing.py",)),
    "MINIMAL_TEST_INCLUDE_AUDIO_JOBS": (_UNDER_PYTEST, ("api/v1/router_groups/minimal.py",)),
    "PYTEST_XDIST_WORKER": (
        "names the per-run temp DB directory on db_path_utils' test-context path only",
        ("core/DB_Management/db_path_utils.py",),
    ),
    "TLDW_TEST_RUN_ID": (
        "names the per-run temp DB directory on db_path_utils' test-context path only",
        ("core/DB_Management/db_path_utils.py",),
    ),
    "POSTGRES_TEST_DATABASE": (
        "content-Postgres connection alias used when TLDW_CONTENT_PG_* is unset; picks the database, "
        "no auth or isolation effect (the global test conftest sets these, so guarding would refuse "
        "every production-path test)",
        ("core/DB_Management/content_backend.py",),
    ),
    "POSTGRES_TEST_DSN": ("see POSTGRES_TEST_DATABASE", ("core/DB_Management/content_backend.py",)),
    "POSTGRES_TEST_HOST": ("see POSTGRES_TEST_DATABASE", ("core/DB_Management/content_backend.py",)),
    "POSTGRES_TEST_PASSWORD": ("see POSTGRES_TEST_DATABASE", ("core/DB_Management/content_backend.py",)),
    "POSTGRES_TEST_PORT": ("see POSTGRES_TEST_DATABASE", ("core/DB_Management/content_backend.py",)),
    "POSTGRES_TEST_USER": ("see POSTGRES_TEST_DATABASE", ("core/DB_Management/content_backend.py",)),
    "SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS": (
        "adds start/end frames to the caller's own sandbox WebSocket stream; no auth or data-boundary effect",
        ("api/v1/endpoints/sandbox.py", "core/config.py"),
    ),
    "TEST_EMBEDDINGS_UNLOAD_TIMEOUT_SECONDS": (
        "timer tuning (idle model unload); no auth or data-boundary effect",
        ("core/Embeddings/Embeddings_Server/Embeddings_Create.py",),
    ),
    "DEEPSEEK_OCR_TEST_COMPRESS": (
        "DeepSeek-OCR's own test_compress inference option",
        ("core/Ingestion_Media_Processing/OCR/backends/deepseek_ocr.py",),
    ),
    "MCP_HEALTHCHECK_DB_WRITE_TEST": (
        _NOT_A_SEAM,
        (
            "core/MCP_unified/modules/implementations/characters_module.py",
            "core/MCP_unified/modules/implementations/chats_module.py",
            "core/MCP_unified/modules/implementations/notes_module.py",
            "core/MCP_unified/modules/implementations/prompts_module.py",
        ),
    ),
    "PROMPT_STUDIO_MAX_TEST_CASES": (_NOT_A_SEAM, ("api/v1/API_Deps/prompt_studio_deps.py",)),
    "CONTESTED": (_NOT_A_SEAM, ("core/Claims_Extraction/claims_engine.py",)),
    "FVA_CONTESTED_THRESHOLD": (_NOT_A_SEAM, ("core/Claims_Extraction/fva_pipeline.py",)),
    "EVALS_ABTEST_JOBS_WORKER_ENABLED": (_NOT_A_SEAM, ("services/startup_notifications_abtest_workers.py",)),
    "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED": (
        _NOT_A_SEAM,
        ("services/startup_notifications_abtest_workers.py",),
    ),
    "EVALS_ABTEST_MAX_ARMS": (_NOT_A_SEAM, ("core/Evaluations/embeddings_abtest_service.py",)),
    "EVALS_ABTEST_MAX_MEDIA_IDS": (_NOT_A_SEAM, ("core/Evaluations/embeddings_abtest_service.py",)),
    "EVALS_ABTEST_MAX_QUERIES": (_NOT_A_SEAM, ("core/Evaluations/embeddings_abtest_service.py",)),
    "EVALS_ABTEST_PERSISTENCE": (_NOT_A_SEAM, ("core/DB_Management/Evaluations_DB.py",)),
    "_TEST_MODE": ("main.py's module global, not an env read", ("main.py",)),
}


def is_guarded(name: str) -> bool:
    return name in GUARDED_TEST_SEAM_KEYS or name.startswith(GUARDED_TEST_SEAM_PREFIXES)


def seam_env_names(source: str, filename: str = "<memory>") -> set[str]:
    """Return the unguarded TEST-shaped names ``source`` uses as string constants."""
    tree = ast.parse(source, filename=filename)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _TEST_NAME.fullmatch(node.value)
        and not is_guarded(node.value)
    }


def current_reads(app_root: Path = APP_ROOT) -> dict[str, set[str]]:
    """env name -> modules (relative to ``app_root``) that read it."""
    reads: dict[str, set[str]] = {}
    for path in sorted(app_root.rglob("*.py")):
        rel = path.relative_to(app_root)
        if "tests" in rel.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "TEST" not in text:
            continue
        for name in seam_env_names(text, rel.as_posix()):
            reads.setdefault(name, set()).add(rel.as_posix())
    return reads


def ratchet_violations(
    reads: dict[str, set[str]], allowlist: dict[str, tuple[str, tuple[str, ...]]]
) -> list[str]:
    """One line per unguarded, unallowlisted (name, module) read, and per stale allowlist pair."""
    allowed = {(name, module) for name, (_reason, modules) in allowlist.items() for module in modules}
    found = {(name, module) for name, modules in reads.items() for module in modules}
    problems = [_unallowlisted(name, module) for name, module in sorted(found - allowed)]
    problems += [
        f"app/{module}: ALLOWLIST pair for {name} is stale (no longer read there). "
        f"Remove it from ALLOWLIST in {THIS_FILE}."
        for name, module in sorted(allowed - found)
    ]
    return problems


def _unallowlisted(name: str, module: str) -> str:
    if not _SEAM_SHAPED.search(name):
        return (
            f"app/{module}: {name} contains TEST only inside a word, so it is not a test seam. "
            f"Allowlist ({name!r}, {module!r}) with reason _NOT_A_SEAM in {THIS_FILE}."
        )
    return (
        f"app/{module}: reads {name}, which validate_test_runtime_flags() does not guard. Add it to "
        f"GUARDED_TEST_SEAM_KEYS in {GUARD_FILE}, or allowlist ({name!r}, {module!r}) with a reason "
        f"in {THIS_FILE} (a reviewed baseline bump)."
    )


def test_every_test_env_read_is_guarded_or_allowlisted() -> None:
    problems = ratchet_violations(current_reads(), ALLOWLIST)
    assert not problems, "\n".join(problems)


def test_every_allowlist_entry_has_a_reason_and_no_guarded_name() -> None:
    assert all(reason.strip() and modules for reason, modules in ALLOWLIST.values())
    assert not [name for name in ALLOWLIST if is_guarded(name)]


@pytest.mark.parametrize(
    ("snippet", "name"),
    [
        ('import os\nx = os.getenv("NEW_TEST_BYPASS")\n', "NEW_TEST_BYPASS"),
        ('from os import environ as e\nx = e["SKIP_AUTH_IN_TESTS"]\n', "SKIP_AUTH_IN_TESTS"),
        ('def f(flag):\n    return flag("PYTEST_CURRENT_TEST")\n', "PYTEST_CURRENT_TEST"),
    ],
)
def test_matcher_sees_seeded_violations(snippet: str, name: str) -> None:
    assert seam_env_names(snippet) == {name}
    problems = ratchet_violations({name: {"core/new.py"}}, {})
    assert len(problems) == 1
    assert problems[0].startswith(f"app/core/new.py: reads {name}, which validate_test_runtime_flags()")
    assert GUARD_FILE in problems[0] and THIS_FILE in problems[0]


def test_a_domain_word_points_to_the_not_a_seam_reason() -> None:
    assert seam_env_names('x = os.getenv("SHOW_LATEST")\n') == {"SHOW_LATEST"}
    (problem,) = ratchet_violations({"SHOW_LATEST": {"core/new.py"}}, {})
    assert "_NOT_A_SEAM" in problem and THIS_FILE in problem
    assert "GUARDED_TEST_SEAM_KEYS" not in problem


def test_matcher_skips_guarded_names_docstrings_and_comments() -> None:
    snippet = (
        '"""Reads NEW_TEST_BYPASS."""\n# os.getenv("OTHER_TEST")\n'
        'a = os.getenv("TEST_MODE")\nb = os.getenv("RG_TEST_FORCE_STUB_RATE")\nc = "a test"\n'
    )
    assert seam_env_names(snippet) == set()


def test_a_new_reader_of_an_allowlisted_name_fails() -> None:
    allowlist: dict[str, tuple[str, tuple[str, ...]]] = {"PYTEST_CURRENT_TEST": ("reason", ("main.py",))}
    reads = {"PYTEST_CURRENT_TEST": {"main.py", "core/new.py"}}
    assert [p.split(":")[0] for p in ratchet_violations(reads, allowlist)] == ["app/core/new.py"]


def test_stale_allowlist_pair_fails() -> None:
    assert ratchet_violations({}, {"GONE_TEST": ("reason", ("main.py",))}) == [
        f"app/main.py: ALLOWLIST pair for GONE_TEST is stale (no longer read there). "
        f"Remove it from ALLOWLIST in {THIS_FILE}."
    ]


def test_scan_finds_seeded_files(tmp_path: Path) -> None:
    files = {
        "core/new_seam.py": 'import os\nX = os.getenv("SKIP_AUTH_IN_TESTS")\n',
        "core/reader.py": "import os\nX = os.environ.get('PYTEST_CURRENT_TEST')\n",
        "core/guarded.py": 'import os\nX = os.getenv("TEST_MODE")\nY = os.getenv("RG_TEST_NEW")\n',
        "core/Thing/tests/test_x.py": 'X = "OTHER_TEST_FLAG"\n',  # test code under app/ is skipped
        "core/docs_only.py": '"""Reads NEW_TEST_FLAG."""\n# os.getenv("NEW_TEST_FLAG")\n',
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    reads = current_reads(tmp_path)

    assert reads == {"SKIP_AUTH_IN_TESTS": {"core/new_seam.py"}, "PYTEST_CURRENT_TEST": {"core/reader.py"}}
    allowlist: dict[str, tuple[str, tuple[str, ...]]] = {"PYTEST_CURRENT_TEST": ("reason", ("main.py",))}
    assert [problem.split(",")[0] for problem in ratchet_violations(reads, allowlist)] == [
        "app/core/reader.py: reads PYTEST_CURRENT_TEST",
        "app/core/new_seam.py: reads SKIP_AUTH_IN_TESTS",
        f"app/main.py: ALLOWLIST pair for PYTEST_CURRENT_TEST is stale (no longer read there). "
        f"Remove it from ALLOWLIST in {THIS_FILE}.",
    ]


def test_backend_required_runs_this_ratchet() -> None:
    steps = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Enforce CI contracts and code ratchets"
    ]
    assert len(steps) == 1
    assert THIS_FILE in steps[0]["run"]


def test_backend_required_runs_the_guard_tests() -> None:
    """AC#7: the startup guard and the AuthNZ refusals run in the required gate."""
    steps = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Enforce CI contracts and code ratchets"
    ]
    assert len(steps) == 1
    assert [path for path in GUARD_TESTS if path not in steps[0]["run"]] == []


if __name__ == "__main__":
    current = current_reads()
    for env_name in sorted(current):
        print(f"{env_name}: {', '.join(sorted(current[env_name]))}")
    print(f"{len(current)} names in {sum(len(m) for m in current.values())} (name, module) pairs")
