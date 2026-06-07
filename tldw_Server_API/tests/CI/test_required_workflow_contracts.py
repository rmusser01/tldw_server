import ast
import fnmatch
import re
from pathlib import Path

import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _get_step(steps: list[dict], name: str) -> dict:
    matching = [step for step in steps if step.get("name") == name]
    assert matching, f"{name} step missing"
    return matching[0]


def _python_test_function_names(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name.startswith("test")
        and not _is_pytest_fixture(node)
    }


def _is_pytest_fixture(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
    return False


def _python_test_node_ids(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node_ids: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name.startswith("test") and not _is_pytest_fixture(node):
                node_ids.add(node.name)
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if (
                isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
                and child.name.startswith("test")
                and not _is_pytest_fixture(child)
            ):
                node_ids.add(f"{node.name}::{child.name}")
    return node_ids


def _assert_ffmpeg_portaudio_setup(path: str, job_name: str) -> None:
    workflow = _load(path)
    steps = workflow["jobs"][job_name]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-portaudio"] == "true"


def _assert_portaudio_installed_before_python_setup(path: str, job_name: str) -> None:
    workflow = _load(path)
    steps = workflow["jobs"][job_name]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    setup_step = _get_step(steps, "Setup Python and backend")
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-portaudio"] == "true"
    assert steps.index(install_step) < steps.index(setup_step)


def test_backend_required_has_noop_and_execute_paths() -> None:
    workflow = _load(".github/workflows/backend-required.yml")
    jobs = workflow["jobs"]
    assert "backend-required" in jobs


def test_backend_required_installs_portaudio_for_pyaudio_builds() -> None:
    _assert_ffmpeg_portaudio_setup(".github/workflows/backend-required.yml", "backend-required")


def test_backend_required_type_checks_only_changed_python_files() -> None:
    workflow = _load(".github/workflows/backend-required.yml")
    steps = workflow["jobs"]["backend-required"]["steps"]
    type_step = _get_step(steps, "Type check changed backend modules")
    assert type_step.get("continue-on-error") is True
    run_script = type_step["run"]
    assert "git diff --name-only" in run_script
    assert "No backend Python files changed; skipping mypy." in run_script
    assert "mypy --follow-imports=silent --ignore-missing-imports" in run_script
    assert "mypy tldw_Server_API/" not in run_script


def test_coverage_required_is_path_conditional() -> None:
    workflow = _load(".github/workflows/coverage-required.yml")
    jobs = workflow["jobs"]
    assert "coverage-required" in jobs


def test_coverage_required_installs_portaudio_for_pyaudio_builds() -> None:
    _assert_ffmpeg_portaudio_setup(".github/workflows/coverage-required.yml", "coverage-required")


def test_coverage_required_uses_documented_global_floor() -> None:
    workflow = _load(".github/workflows/coverage-required.yml")
    steps = workflow["jobs"]["coverage-required"]["steps"]
    coverage_step = _get_step(steps, "Run global coverage floor")
    assert "--cov-fail-under=5" in coverage_step["run"]


def test_frontend_required_lane_exists() -> None:
    workflow = _load(".github/workflows/frontend-required.yml")
    jobs = workflow["jobs"]
    assert "frontend-required" in jobs


def test_frontend_required_does_not_require_missing_lockfile_cache() -> None:
    workflow = _load(".github/workflows/frontend-required.yml")
    steps = workflow["jobs"]["frontend-required"]["steps"]

    setup_node = _get_step(steps, "Setup Node.js")
    setup_with = setup_node.get("with") or {}
    cache_dependency_path = setup_with.get("cache-dependency-path")
    if cache_dependency_path and not Path(str(cache_dependency_path)).exists():
        raise AssertionError(
            f"frontend-required references missing cache dependency path: {cache_dependency_path}"
        )

    setup_bun = _get_step(steps, "Setup Bun")
    if setup_bun.get("uses") != "oven-sh/setup-bun@v2":
        raise AssertionError("frontend-required must configure Bun with oven-sh/setup-bun@v2")

    install_step = _get_step(steps, "Install frontend dependencies")
    if install_step.get("working-directory") != "apps":
        raise AssertionError("frontend-required must install workspace dependencies from apps/")
    run_script = str(install_step.get("run") or "")
    if "bun install" not in run_script:
        raise AssertionError("frontend-required must install dependencies with bun install")
    if "npm ci" in run_script:
        raise AssertionError("frontend-required should not use npm ci for Bun workspace dependencies")

    test_step = _get_step(steps, "Run frontend unit tests")
    test_run_script = str(test_step.get("run") or "")
    if "bunx vitest run --changed=" not in test_run_script:
        raise AssertionError("frontend-required unit tests must use changed-only vitest execution in PRs")
    if "bun run test:run" not in test_run_script:
        raise AssertionError("frontend-required must keep full-suite fallback when base SHA is unavailable")


def test_e2e_required_lane_exists_and_is_conditional() -> None:
    workflow = _load(".github/workflows/e2e-required.yml")
    jobs = workflow["jobs"]
    assert "e2e-required" in jobs


def test_e2e_required_installs_portaudio_for_pyaudio_builds() -> None:
    _assert_ffmpeg_portaudio_setup(".github/workflows/e2e-required.yml", "e2e-required")


def test_e2e_required_explicitly_loads_pytest_asyncio_plugin() -> None:
    workflow = _load(".github/workflows/e2e-required.yml")
    steps = workflow["jobs"]["e2e-required"]["steps"]
    primary = _get_step(steps, "Run critical e2e suite (attempt 1)")
    retry = _get_step(steps, "Run critical e2e suite (retry)")
    assert "-p pytest_asyncio.plugin" in primary["run"]
    assert "-p pytest_asyncio.plugin" in retry["run"]


def test_frontend_e2e_tiers_install_portaudio_before_backend_dependency_setup() -> None:
    for job_name in ("critical", "features", "admin"):
        _assert_portaudio_installed_before_python_setup(
            ".github/workflows/frontend-e2e-tiers.yml",
            job_name,
        )


def test_security_required_lane_exists_and_uses_threshold_policy() -> None:
    workflow = _load(".github/workflows/security-required.yml")
    jobs = workflow["jobs"]
    assert "security-required" in jobs


def test_required_gate_names_documented() -> None:
    text = Path("Docs/Development/CI_REQUIRED_GATES.md").read_text(encoding="utf-8")
    for check_name in [
        "backend-required",
        "security-required",
        "coverage-required",
        "frontend-required",
        "e2e-required",
    ]:
        assert check_name in text


def test_security_required_bandit_does_not_preempt_threshold_filter() -> None:
    workflow = _load(".github/workflows/security-required.yml")
    steps = workflow["jobs"]["security-required"]["steps"]
    bandit_steps = [step for step in steps if step.get("name") == "Run Bandit scan"]
    assert bandit_steps, "Run Bandit scan step missing"
    assert "--exit-zero" in bandit_steps[0]["run"]


def test_security_required_includes_dependency_review_gate() -> None:
    workflow = _load(".github/workflows/security-required.yml")
    steps = workflow["jobs"]["security-required"]["steps"]
    dep_review_steps = [step for step in steps if step.get("name") == "Dependency review (high/critical)"]
    assert dep_review_steps, "Dependency review step missing"
    assert re.match(r"^actions/dependency-review-action@[0-9a-f]{40}$", dep_review_steps[0]["uses"])


def test_legacy_ci_workflow_name_remains_stable_for_branch_protection() -> None:
    workflow = _load(".github/workflows/ci.yml")
    assert workflow["name"] == "CI"


def test_full_suite_pytest_steps_do_not_leak_shared_postgres_dsn() -> None:
    workflow = _load(".github/workflows/ci.yml")
    jobs = workflow["jobs"]
    step_names = {
        "Run Python 3.11 compatibility smoke tests",
        "Run shard tests",
        "Run OS shard tests",
        "Run release OS shard tests",
        "Run legacy-free media checks",
    }

    checked = 0
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("name") not in step_names:
                continue
            run_script = str(step.get("run") or "")
            assert 'export DATABASE_URL="sqlite:///./Databases/users.db"' in run_script
            assert "unset TEST_DATABASE_URL POSTGRES_TEST_DB" in run_script
            checked += 1

    assert checked >= len(step_names)


def test_embedding_model_cache_restore_is_non_blocking() -> None:
    workflow = _load(".github/workflows/ci.yml")
    cache_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Cache embedding models"
    ]

    assert len(cache_steps) == 5
    for step in cache_steps:
        assert step.get("uses") == "actions/cache@v5"
        assert step.get("continue-on-error") is True
        assert "github.event_name != 'workflow_dispatch'" in str(step.get("if"))


def test_full_suite_test_result_uploads_are_non_blocking() -> None:
    workflow = _load(".github/workflows/ci.yml")
    upload_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("name", "")).startswith("Upload test results")
    ]

    assert len(upload_steps) == 6
    for step in upload_steps:
        assert step.get("if") == "always()"
        assert step.get("continue-on-error") is True
        assert step.get("uses") == "actions/upload-artifact@v7"


def test_linux_311_smoke_is_sharded_for_timeout_control() -> None:
    workflow = _load(".github/workflows/ci.yml")
    job = workflow["jobs"]["full-suite-linux-311-smoke"]
    assert job["name"] == "Full Suite (Ubuntu / Python 3.11 / ${{ matrix.shard.name }})"

    shards = job["strategy"]["matrix"]["shard"]
    shard_paths = {shard["name"]: set(str(shard["paths"]).split()) for shard in shards}
    assert set(shard_paths) == {"authnz-unit", "config", "core", "utils-http"}
    assert shard_paths["authnz-unit"] == {"tldw_Server_API/tests/AuthNZ_Unit"}
    assert shard_paths["config"] == {"tldw_Server_API/tests/Config"}
    assert shard_paths["utils-http"] == {
        "tldw_Server_API/tests/Utils",
        "tldw_Server_API/tests/http_client",
    }
    assert {
        "tldw_Server_API/tests/test_*.py",
        "tldw_Server_API/tests/Health",
        "tldw_Server_API/tests/sanity_tests",
        "tldw_Server_API/tests/schemas",
        "tldw_Server_API/tests/unit",
    } == shard_paths["core"]

    steps = job["steps"]
    for step_name in [
        "Smoke start server (single-user)",
        "Smoke health check",
        "Print smoke server log on failure",
        "Smoke stop server",
    ]:
        step = _get_step(steps, step_name)
        assert "matrix.shard.name == 'core'" in step["if"]

    run_step = _get_step(steps, "Run Python 3.11 compatibility smoke tests")
    run_script = run_step["run"]
    assert "${{ matrix.shard.paths }}" in run_script
    assert "test-results-linux-3.11-smoke" not in run_script


def test_full_suite_splits_slow_chat_and_retrieval_shards() -> None:
    workflow = _load(".github/workflows/ci.yml")
    matrix_jobs = [
        "full-suite-linux-312-shards",
        "full-suite-linux-313-shards",
        "full-suite-macos-312-shards",
        "full-suite-windows-312-shards",
        "full-suite-os-313-release-shards",
    ]

    for job_name in matrix_jobs:
        shards = workflow["jobs"][job_name]["strategy"]["matrix"]["shard"]
        shard_names = {shard["name"] for shard in shards}
        rag_new_shards = {
            "rag-new-integration-agentic",
            "rag-new-integration-batch",
            "rag-new-integration-core",
            "rag-new-integration-research",
            "rag-new-property-core",
            "rag-new-unit-agentic",
            "rag-new-unit-cache-vector",
            "rag-new-unit-core-misc",
            "rag-new-unit-guardrails-source",
            "rag-new-unit-media-ingest",
            "rag-new-unit-pipeline",
            "rag-new-unit-rag-contracts",
            "rag-new-unit-unified-pipeline",
        }
        assert "ai-retrieval" not in shard_names
        assert "chat-llm" not in shard_names
        assert "product-modules" not in shard_names
        assert "chat-character-unit" not in shard_names
        assert "rag-research" not in shard_names
        assert "rag-new" not in shard_names
        assert "core-smoke" not in shard_names
        assert "admin-a-b" not in shard_names
        assert "admin-c-d" not in shard_names
        assert "admin-e-l" not in shard_names
        assert "admin-m-r" not in shard_names
        assert "admin-s-z" not in shard_names
        assert "admin-bundle" not in shard_names
        assert "admin-e2e" not in shard_names
        assert "ai-chromadb-chunking" not in shard_names
        assert "auth-db" not in shard_names
        assert "ai-embeddings" not in shard_names
        assert "ai-embeddings-policy-v5" not in shard_names
        assert "chat-character-legacy" not in shard_names
        assert "chat-character-integration" not in shard_names
        assert "product-claims" not in shard_names
        assert "product-evaluations" not in shard_names
        assert "product-prompts-workflows" not in shard_names
        assert "platform-mcp" not in shard_names
        assert {
            "core-audit-security",
            "core-config",
            "core-server-smoke",
            "core-setup-usage",
            "core-utils-tooling",
        }.issubset(shard_names)
        assert {
            "admin-a-api",
            "admin-backup-api",
            "admin-backup-core",
            "admin-budgets",
            "admin-bundle-ops",
            "admin-bundle-sanitizers",
            "admin-byok-core",
            "admin-byok-validation",
            "admin-conflicts-data-admin",
            "admin-data-ops",
            "admin-data-subject-api",
            "admin-data-subject-repo-dsr",
            "admin-e2e-access",
            "admin-e2e-reset-backups",
            "admin-e2e-seed",
            "admin-e2e-session-dsr",
            "admin-e2e-single-user",
            "admin-g-i",
            "admin-incidents",
            "admin-llm-providers",
            "admin-llm-usage",
            "admin-maintenance-misc",
            "admin-monitoring",
            "admin-ops-dependencies",
            "admin-ops-endpoints",
            "admin-ops-webhooks-orgs",
            "admin-pricing-retention-roles-router",
            "admin-profiles-rate-registration",
            "admin-s-sessions-settings",
            "admin-split-storage-tools",
            "admin-system-usage",
            "admin-users",
            "admin-watchlists-webhooks",
        }.issubset(shard_names)
        auth_db_shards = {
            "auth-core",
            "auth-sqlite",
            "auth-unit-a-l",
            "auth-unit-m-z",
            "chacha-core-stores",
            "chacha-character-conversation",
            "chacha-content-persona",
            "db-privileges",
        }
        assert auth_db_shards.issubset(shard_names)
        assert {
            "ai-chromadb",
            "ai-chunking-code-json-xml",
            "ai-chunking-core",
            "ai-chunking-semantic-security",
            "ai-chunking-templates",
            "ai-embeddings-core",
            "ai-embeddings-dlq-config",
            "ai-embeddings-hyde-ledger",
            "ai-embeddings-jobs-runtime",
            "ai-embeddings-media-validation",
            "ai-embeddings-observability",
            "ai-embeddings-policy",
            "ai-embeddings-v5-core",
            "ai-embeddings-v5-integration",
            "vector-stores",
            "paper-search",
            "rag-legacy",
            "research-websearch",
        }.issubset(shard_names)
        assert rag_new_shards.issubset(shard_names)
        assert {
            "chat-character-legacy-core",
            "chat-character-legacy-files",
            "chat-character-legacy-worldbook",
            "chat-character-db",
            "chat-character-unit-core",
            "chat-character-unit-chat",
            "chat-character-unit-persona",
            "chat-character-unit-prd",
            "chat-character-property",
            "chat-character-integration-api",
            "chat-character-integration-chat",
            "chat-character-integration-context",
            "chat-core",
            "llm-adapters-unit",
            "llm-adapters-chat-endpoint",
            "llm-adapters-chat-errors-core",
            "llm-adapters-chat-errors-extra",
            "llm-adapters-orchestrator-core",
            "llm-adapters-orchestrator-extra",
            "llm-calls-core",
            "llm-calls-property",
            "llm-local-runtime",
            "llm-local-backends",
        }.issubset(shard_names)
        assert {
            "product-claims-core",
            "product-claims-engine",
            "product-claims-monitoring",
            "product-claims-service",
            "product-collections",
            "product-evaluations-abtest",
            "product-evaluations-core",
            "product-evaluations-integration",
            "product-evaluations-recipes-persona",
            "product-evaluations-unit",
            "product-flashcards",
            "product-notes-persona",
            "product-prompt-studio",
            "product-prompts-legacy",
            "product-prompts-new",
            "product-watchlists",
            "product-workflows",
        }.issubset(shard_names)
        platform_shards = {
            "platform-infrastructure-metrics",
            "platform-mcp-core",
            "platform-resource-governance",
            "platform-sandbox-admin-artifacts",
            "platform-sandbox-runtimes",
            "platform-sandbox-state-store",
            "platform-sandbox-ws-streams",
            "platform-services-core",
            "platform-services-shutdown-lifespan",
            "platform-services-startup",
        }
        assert platform_shards.issubset(shard_names)
        shard_paths = {shard["name"]: shard["paths"] for shard in shards}
        shard_path_sets = {
            name: set(str(paths).split()) for name, paths in shard_paths.items()
        }
        assert "tldw_Server_API/tests/VectorStores" not in shard_path_sets["ai-embeddings-core"]
        assert shard_path_sets["ai-chromadb"] == {"tldw_Server_API/tests/ChromaDB"}
        assert "tldw_Server_API/tests/Admin" not in shard_path_sets["core-server-smoke"]
        assert "tldw_Server_API/tests/RAG_NEW" not in shard_path_sets["rag-legacy"]
        for shard_name in rag_new_shards:
            assert "tldw_Server_API/tests/RAG" not in shard_path_sets[shard_name]
        assert "tldw_Server_API/tests/PaperSearch" not in shard_path_sets["research-websearch"]
        assert "tldw_Server_API/tests/WebSearch" not in shard_path_sets["paper-search"]
        assert "tldw_Server_API/tests/Characters" not in shard_path_sets["chat-character-legacy-core"]
        assert "tldw_Server_API/tests/Character_Chat_NEW/unit" not in shard_path_sets["chat-character-property"]
        assert "tldw_Server_API/tests/Claims" not in shard_path_sets["product-collections"]
        assert "tldw_Server_API/tests/Evaluations" not in shard_path_sets["product-claims-core"]

        core_shards = {
            "core-audit-security",
            "core-config",
            "core-server-smoke",
            "core-setup-usage",
            "core-utils-tooling",
        }
        core_dirs = (
            "Audit",
            "Config",
            "Health",
            "Helper_Scripts",
            "Logging",
            "Security",
            "Setup",
            "Usage",
            "Utils",
            "helpers",
            "http_client",
            "lint",
            "sanity_tests",
            "schemas",
            "unit",
        )
        core_files = {
            str(path)
            for path in Path("tldw_Server_API/tests").glob("test*.py")
        }
        for dirname in core_dirs:
            core_files.update(
                str(path)
                for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
            )

        covered_core_files: dict[str, str] = {}
        for shard_name in core_shards:
            for pattern in shard_path_sets[shard_name]:
                if Path(pattern).is_dir():
                    prefix = f"{pattern.rstrip('/')}/"
                    matches = {
                        filename
                        for filename in core_files
                        if filename.startswith(prefix)
                    }
                else:
                    matches = {
                        filename
                        for filename in core_files
                        if fnmatch.fnmatch(filename, pattern)
                    }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_core_files, (
                        f"{filename} matched both "
                        f"{covered_core_files[filename]} and {shard_name}"
                    )
                    covered_core_files[filename] = shard_name

        assert set(covered_core_files) == core_files

        admin_shards = {
            "admin-a-api",
            "admin-backup-api",
            "admin-backup-core",
            "admin-budgets",
            "admin-bundle-ops",
            "admin-bundle-sanitizers",
            "admin-byok-core",
            "admin-byok-validation",
            "admin-conflicts-data-admin",
            "admin-data-ops",
            "admin-data-subject-api",
            "admin-data-subject-repo-dsr",
            "admin-e2e-access",
            "admin-e2e-reset-backups",
            "admin-e2e-seed",
            "admin-e2e-session-dsr",
            "admin-e2e-single-user",
            "admin-g-i",
            "admin-incidents",
            "admin-llm-providers",
            "admin-llm-usage",
            "admin-maintenance-misc",
            "admin-monitoring",
            "admin-ops-dependencies",
            "admin-ops-endpoints",
            "admin-ops-webhooks-orgs",
            "admin-pricing-retention-roles-router",
            "admin-profiles-rate-registration",
            "admin-s-sessions-settings",
            "admin-split-storage-tools",
            "admin-system-usage",
            "admin-users",
            "admin-watchlists-webhooks",
        }
        admin_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Admin").glob("test*.py")
        }
        admin_test_functions_by_file = {
            filename: _python_test_function_names(filename)
            for filename in admin_files
        }
        covered_admin_files: dict[str, str] = {}
        covered_admin_nodeids: set[str] = set()
        covered_admin_nodeids_by_file: dict[str, set[str]] = {}
        for shard_name in admin_shards:
            for pattern in shard_path_sets[shard_name]:
                file_pattern = pattern.split("::", 1)[0]
                assert file_pattern.startswith("tldw_Server_API/tests/Admin/")
                matches = {
                    filename
                    for filename in admin_files
                    if fnmatch.fnmatch(filename, file_pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                if "::" in pattern:
                    assert len(matches) == 1, (
                        f"{shard_name} node id must target one file: {pattern}"
                    )
                    test_name = pattern.rsplit("::", 1)[1]
                    filename = next(iter(matches))
                    assert test_name in admin_test_functions_by_file[filename], (
                        f"{shard_name} references missing test node id: {pattern}"
                    )
                    assert pattern not in covered_admin_nodeids, (
                        f"{pattern} is listed by multiple admin e2e shards"
                    )
                    covered_admin_nodeids.add(pattern)
                    covered_admin_nodeids_by_file.setdefault(filename, set()).add(
                        test_name
                    )
                for filename in matches:
                    if filename in covered_admin_files and "::" not in pattern:
                        raise AssertionError(
                            f"{filename} matched both "
                            f"{covered_admin_files[filename]} and {shard_name}"
                        )
                    covered_admin_files[filename] = shard_name

        assert set(covered_admin_files) == admin_files
        for filename, covered_nodeids in covered_admin_nodeids_by_file.items():
            assert covered_nodeids == admin_test_functions_by_file[filename], (
                f"{filename} admin node-id shard coverage mismatch"
            )

        auth_db_dirs = (
            "AuthNZ",
            "AuthNZ_Postgres",
            "AuthNZ_SQLite",
            "AuthNZ_Unit",
            "ChaChaNotesDB",
            "DB",
            "DB_Management",
            "MediaDB2",
            "PrivilegeCatalog",
            "Privileges",
        )
        auth_db_roots = {
            f"tldw_Server_API/tests/{dirname}/"
            for dirname in auth_db_dirs
        }
        auth_db_files = {
            str(path)
            for dirname in auth_db_dirs
            for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
        }
        covered_auth_db_files: dict[str, str] = {}
        for shard_name in auth_db_shards:
            for pattern in shard_path_sets[shard_name]:
                assert any(
                    pattern == root.rstrip("/") or pattern.startswith(root)
                    for root in auth_db_roots
                )
                if Path(pattern).is_dir():
                    prefix = f"{pattern.rstrip('/')}/"
                    matches = {
                        filename
                        for filename in auth_db_files
                        if filename.startswith(prefix)
                    }
                else:
                    matches = {
                        filename
                        for filename in auth_db_files
                        if fnmatch.fnmatch(filename, pattern)
                    }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_auth_db_files, (
                        f"{filename} matched both "
                        f"{covered_auth_db_files[filename]} and {shard_name}"
                    )
                    covered_auth_db_files[filename] = shard_name

        assert set(covered_auth_db_files) == auth_db_files

        chunking_shards = {
            "ai-chunking-code-json-xml",
            "ai-chunking-core",
            "ai-chunking-semantic-security",
            "ai-chunking-templates",
            "ai-chunking-templates-api",
            "ai-chunking-templates-integration",
        }
        chunking_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Chunking").glob("**/test*.py")
        }
        chunking_test_nodeids_by_file = {
            filename: _python_test_node_ids(filename)
            for filename in chunking_files
        }
        covered_chunking_files: dict[str, str] = {}
        covered_chunking_nodeids_by_file: dict[str, set[str]] = {}
        for shard_name in chunking_shards:
            for pattern in shard_path_sets[shard_name]:
                file_pattern = pattern.split("::", 1)[0]
                assert pattern.startswith("tldw_Server_API/tests/Chunking/")
                matches = {
                    filename
                    for filename in chunking_files
                    if fnmatch.fnmatch(filename, file_pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                if "::" in pattern:
                    assert len(matches) == 1, (
                        f"{shard_name} node id must target one file: {pattern}"
                    )
                    node_id = pattern.split("::", 1)[1]
                    filename = next(iter(matches))
                    matched_nodeids = {
                        known_nodeid
                        for known_nodeid in chunking_test_nodeids_by_file[filename]
                        if known_nodeid == node_id or known_nodeid.startswith(f"{node_id}::")
                    }
                    assert matched_nodeids, (
                        f"{shard_name} references missing chunking test node id: {pattern}"
                    )
                    already_covered = (
                        covered_chunking_nodeids_by_file.setdefault(filename, set())
                        & matched_nodeids
                    )
                    assert not already_covered, (
                        f"{filename} node ids matched multiple chunking shards: "
                        f"{sorted(already_covered)}"
                    )
                    covered_chunking_nodeids_by_file[filename].update(matched_nodeids)
                for filename in matches:
                    if "::" not in pattern:
                        assert filename not in covered_chunking_files, (
                            f"{filename} matched both "
                            f"{covered_chunking_files[filename]} and {shard_name}"
                        )
                    covered_chunking_files.setdefault(filename, shard_name)

        assert set(covered_chunking_files) == chunking_files
        for filename, covered_nodeids in covered_chunking_nodeids_by_file.items():
            assert covered_nodeids == chunking_test_nodeids_by_file[filename], (
                f"{filename} chunking node-id shard coverage mismatch"
            )

        embedding_shards = {
            "ai-embeddings-core",
            "ai-embeddings-dlq-config",
            "ai-embeddings-hyde-ledger",
            "ai-embeddings-jobs-runtime",
            "ai-embeddings-media-validation",
            "ai-embeddings-observability",
            "ai-embeddings-policy",
            "ai-embeddings-v5-core",
            "ai-embeddings-v5-integration",
        }
        embedding_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Embeddings").glob("test*.py")
        }
        covered_embedding_files: dict[str, str] = {}
        for shard_name in embedding_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Embeddings/")
                matches = {
                    filename
                    for filename in embedding_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_embedding_files, (
                        f"{filename} matched both "
                        f"{covered_embedding_files[filename]} and {shard_name}"
                    )
                    covered_embedding_files[filename] = shard_name

        assert set(covered_embedding_files) == embedding_files

        rag_new_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/RAG_NEW").glob("**/test*.py")
        }
        covered_rag_new_files: dict[str, str] = {}
        for shard_name in rag_new_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/RAG_NEW/")
                matches = {
                    filename
                    for filename in rag_new_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_rag_new_files, (
                        f"{filename} matched both "
                        f"{covered_rag_new_files[filename]} and {shard_name}"
                    )
                    covered_rag_new_files[filename] = shard_name

        assert set(covered_rag_new_files) == rag_new_files

        platform_dirs = (
            "Infrastructure",
            "MCP",
            "MCP_unified",
            "Metrics",
            "Monitoring",
            "Resource_Governance",
            "Services",
            "sandbox",
        )
        platform_roots = {
            f"tldw_Server_API/tests/{dirname}/"
            for dirname in platform_dirs
        }
        platform_files = {
            str(path)
            for dirname in platform_dirs
            for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
        }
        covered_platform_files: dict[str, str] = {}
        for shard_name in platform_shards:
            for pattern in shard_path_sets[shard_name]:
                assert any(
                    pattern == root.rstrip("/") or pattern.startswith(root)
                    for root in platform_roots
                )
                if Path(pattern).is_dir():
                    prefix = f"{pattern.rstrip('/')}/"
                    matches = {
                        filename
                        for filename in platform_files
                        if filename.startswith(prefix)
                    }
                else:
                    matches = {
                        filename
                        for filename in platform_files
                        if fnmatch.fnmatch(filename, pattern)
                    }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_platform_files, (
                        f"{filename} matched both "
                        f"{covered_platform_files[filename]} and {shard_name}"
                    )
                    covered_platform_files[filename] = shard_name

        assert set(covered_platform_files) == platform_files

        legacy_character_shards = {
            "chat-character-legacy-core",
            "chat-character-legacy-files",
            "chat-character-legacy-worldbook",
        }
        legacy_character_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Character_Chat").glob("**/test*.py")
        }
        covered_legacy_character_files: dict[str, str] = {}
        for shard_name in legacy_character_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Character_Chat/")
                matches = {
                    filename
                    for filename in legacy_character_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_legacy_character_files, (
                        f"{filename} matched both "
                        f"{covered_legacy_character_files[filename]} and {shard_name}"
                    )
                    covered_legacy_character_files[filename] = shard_name

        assert set(covered_legacy_character_files) == legacy_character_files

        new_character_integration_shards = {
            "chat-character-integration-api",
            "chat-character-integration-chat",
            "chat-character-integration-context",
        }
        new_character_integration_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Character_Chat_NEW/integration").glob("test*.py")
        }
        covered_new_character_integration_files: dict[str, str] = {}
        for shard_name in new_character_integration_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Character_Chat_NEW/integration/")
                matches = {
                    filename
                    for filename in new_character_integration_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_new_character_integration_files, (
                        f"{filename} matched both "
                        f"{covered_new_character_integration_files[filename]} and {shard_name}"
                    )
                    covered_new_character_integration_files[filename] = shard_name

        assert set(covered_new_character_integration_files) == new_character_integration_files

        claims_shards = {
            "product-claims-core",
            "product-claims-engine",
            "product-claims-monitoring",
            "product-claims-service",
        }
        claims_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Claims").glob("test*.py")
        }
        covered_claims_files: dict[str, str] = {}
        for shard_name in claims_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Claims/")
                matches = {
                    filename
                    for filename in claims_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_claims_files, (
                        f"{filename} matched both "
                        f"{covered_claims_files[filename]} and {shard_name}"
                    )
                    covered_claims_files[filename] = shard_name

        assert set(covered_claims_files) == claims_files

        evaluation_shards = {
            "product-evaluations-abtest",
            "product-evaluations-core",
            "product-evaluations-integration",
            "product-evaluations-recipes-persona",
            "product-evaluations-unit",
        }
        evaluation_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Evaluations").glob("**/test*.py")
        }
        covered_evaluation_files: dict[str, str] = {}
        for shard_name in evaluation_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Evaluations/")
                matches = {
                    filename
                    for filename in evaluation_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_evaluation_files, (
                        f"{filename} matched both "
                        f"{covered_evaluation_files[filename]} and {shard_name}"
                    )
                    covered_evaluation_files[filename] = shard_name

        assert set(covered_evaluation_files) == evaluation_files

        flashcard_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Flashcards").glob("test*.py")
        }
        covered_flashcard_files = {
            filename
            for pattern in shard_path_sets["product-flashcards"]
            for filename in flashcard_files
            if fnmatch.fnmatch(filename, pattern)
        }

        assert covered_flashcard_files == flashcard_files
