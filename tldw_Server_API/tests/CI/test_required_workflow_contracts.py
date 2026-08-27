import ast
import fnmatch
import json
import re
from pathlib import Path

import pytest
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


def _assert_setup_skips_ffmpeg_but_keeps_portaudio(
    path: str,
    job_name: str,
    step_name: str = "Install FFmpeg and PortAudio (Linux)",
) -> None:
    workflow = _load(path)
    install_step = _get_step(workflow["jobs"][job_name]["steps"], step_name)
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-ffmpeg"] == "false"
    assert install_step["with"]["install-portaudio"] == "true"


def test_ci_postgres_url_exports_are_masked_before_env_write() -> None:
    workflow = _load(".github/workflows/ci.yml")
    export_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Export PG env vars"
    ]
    assert export_steps, "Export PG env vars steps missing"
    for step in export_steps:
        run_script = step["run"]
        password_mask_index = run_script.index('echo "::add-mask::${DB_PASSWORD}"')
        url_mask_index = run_script.index('echo "::add-mask::${DB_URL}"')
        env_write_index = run_script.index('echo "TEST_DATABASE_URL=${DB_URL}"')
        assert password_mask_index < env_write_index
        assert url_mask_index < env_write_index


def test_backend_required_has_noop_and_execute_paths() -> None:
    workflow = _load(".github/workflows/backend-required.yml")
    jobs = workflow["jobs"]
    assert "backend-required" in jobs


def test_backend_required_installs_portaudio_for_pyaudio_builds() -> None:
    _assert_ffmpeg_portaudio_setup(".github/workflows/backend-required.yml", "backend-required")
    workflow = _load(".github/workflows/backend-required.yml")
    steps = workflow["jobs"]["backend-required"]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    assert install_step["with"]["install-ffmpeg"] == "false"


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
    workflow = _load(".github/workflows/coverage-required.yml")
    steps = workflow["jobs"]["coverage-required"]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-ffmpeg"] == "false"
    assert install_step["with"]["install-portaudio"] == "true"


def test_coverage_required_uses_documented_global_floor() -> None:
    workflow = _load(".github/workflows/coverage-required.yml")
    steps = workflow["jobs"]["coverage-required"]["steps"]
    coverage_step = _get_step(steps, "Run global coverage floor")
    assert "--cov-fail-under=12" in coverage_step["run"]


def test_coverage_required_enforces_authnz_coverage_floor() -> None:
    workflow = _load(".github/workflows/coverage-required.yml")
    steps = workflow["jobs"]["coverage-required"]["steps"]
    authnz_step = _get_step(steps, "AuthNZ coverage floor")
    assert "--cov-fail-under=35" in authnz_step["run"]


def test_frontend_required_lane_exists() -> None:
    workflow = _load(".github/workflows/frontend-required.yml")
    jobs = workflow["jobs"]
    assert "frontend-required" in jobs


@pytest.mark.unit
def test_frontend_required_uses_isolated_vitest_shards() -> None:
    workflow = _load(".github/workflows/frontend-required.yml")
    jobs = workflow["jobs"]
    unit_job = jobs["frontend-unit-tests"]
    steps = unit_job["steps"]

    assert unit_job["needs"] == ["changes", "admission"]
    assert unit_job["timeout-minutes"] == 60
    assert unit_job["strategy"] == {
        "fail-fast": False,
        "max-parallel": 8,
        "matrix": {"shard": list(range(1, 9))},
    }

    setup_node = _get_step(steps, "Setup Node.js")
    setup_with = setup_node.get("with") or {}
    cache_dependency_path = setup_with.get("cache-dependency-path")
    if cache_dependency_path and not Path(str(cache_dependency_path)).exists():
        raise AssertionError(
            f"frontend-unit-tests references missing cache dependency path: {cache_dependency_path}"
        )

    setup_bun = _get_step(steps, "Setup Bun")
    if setup_bun.get("uses") != "oven-sh/setup-bun@v2":
        raise AssertionError("frontend-unit-tests must configure Bun with oven-sh/setup-bun@v2")

    install_step = _get_step(steps, "Install frontend dependencies")
    if install_step.get("working-directory") != "apps":
        raise AssertionError("frontend-unit-tests must install workspace dependencies from apps/")
    run_script = str(install_step.get("run") or "")
    if "bun install" not in run_script:
        raise AssertionError("frontend-unit-tests must install dependencies with bun install")
    if "npm ci" in run_script:
        raise AssertionError("frontend-unit-tests should not use npm ci for Bun workspace dependencies")

    assert unit_job["env"]["NODE_OPTIONS"] == "--max-old-space-size=5120"

    test_step = _get_step(steps, "Run package-owned frontend unit tests")
    assert test_step["working-directory"] == "apps"
    test_run_script = str(test_step.get("run") or "")
    assert 'local head_package_root="${GITHUB_WORKSPACE}/${package_repo_path}"' in test_run_script
    assert 'head_command+=("--exclude=${exclude_pattern}")' in test_run_script
    assert '"${head_command[@]}"' in test_run_script
    assert 'bunx vitest run "${failed_files[@]}"' in test_run_script
    assert 'frontend_status=$?' in test_run_script
    assert 'ui_status=$?' in test_run_script
    assert 'if (( frontend_status != 0 || ui_status != 0 )); then' in test_run_script
    assert (
        'git diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA"'
        in test_run_script
    )
    assert (
        'git worktree add --detach "$worktree_path" "$BASE_SHA"'
        in test_run_script
    )
    assert "bun install --frozen-lockfile" in test_run_script
    assert 'run_package "frontend" "apps/tldw-frontend" "../packages/ui/**"' in test_run_script
    assert 'run_package "ui" "apps/packages/ui" ""' in test_run_script
    assert '"--reporter=default"' in test_run_script
    assert '"--reporter=json"' in test_run_script
    assert '"--outputFile.json=${head_report}"' in test_run_script
    assert 'RATCHET_SCRIPT="${GITHUB_WORKSPACE}/Helper_Scripts/ci/vitest_base_ratchet.py"' in test_run_script
    assert 'python3 "$RATCHET_SCRIPT" validate-success' in test_run_script
    assert 'python3 "$RATCHET_SCRIPT" extract' in test_run_script
    assert 'python3 "$RATCHET_SCRIPT" compare' in test_run_script
    assert 'if (( head_status == 0 )); then' in test_run_script
    assert '--changed-files "$CHANGED_FILES_PATH"' in test_run_script
    assert '"--changed=${BASE_SHA}"' in test_run_script
    assert '"--shard=${{ matrix.shard }}/8"' in test_run_script
    assert '"--maxWorkers=1"' in test_run_script
    assert '"--passWithNoTests"' in test_run_script
    assert "bun run test:run" not in test_run_script

    frontend_config = Path("apps/tldw-frontend/vitest.config.ts").read_text(
        encoding="utf-8"
    )
    assert "../packages/ui/src/**/__tests__" in frontend_config

    apps_package = json.loads(Path("apps/package.json").read_text(encoding="utf-8"))
    frontend_package = json.loads(
        Path("apps/tldw-frontend/package.json").read_text(encoding="utf-8")
    )
    ui_package = json.loads(
        Path("apps/packages/ui/package.json").read_text(encoding="utf-8")
    )
    assert {
        apps_package["dependencies"]["jsdom"],
        frontend_package["devDependencies"]["jsdom"],
        ui_package["devDependencies"]["jsdom"],
    } == {"^28.1.0"}

    final_job = jobs["frontend-required"]
    assert final_job["needs"] == ["changes", "admission", "frontend-unit-tests"]
    assert final_job["timeout-minutes"] == 120
    final_steps = final_job["steps"]
    assert not any(step.get("name") == "Run frontend unit tests" for step in final_steps)
    shard_guard = _get_step(final_steps, "Require frontend unit shard success")
    shard_guard_script = str(shard_guard.get("run") or "")
    assert shard_guard["env"] == {
        "TLDW_FRONTEND_CHANGED": "${{ needs.changes.outputs.tldw_frontend_changed }}",
        "UNIT_SHARDS_RESULT": "${{ needs.frontend-unit-tests.result }}",
    }
    assert "$TLDW_FRONTEND_CHANGED" in shard_guard_script
    assert "$UNIT_SHARDS_RESULT" in shard_guard_script
    assert '"$TLDW_FRONTEND_CHANGED" == "false"' in shard_guard_script
    assert '"$TLDW_FRONTEND_CHANGED" != "true"' not in shard_guard_script
    assert "exit 1" in shard_guard_script


def test_e2e_required_lane_exists_and_is_conditional() -> None:
    workflow = _load(".github/workflows/e2e-required.yml")
    jobs = workflow["jobs"]
    assert "e2e-required" in jobs


def test_e2e_required_installs_portaudio_for_pyaudio_builds() -> None:
    _assert_ffmpeg_portaudio_setup(".github/workflows/e2e-required.yml", "e2e-required")
    _assert_setup_skips_ffmpeg_but_keeps_portaudio(
        ".github/workflows/e2e-required.yml",
        "e2e-required",
    )


def test_e2e_required_explicitly_loads_pytest_asyncio_plugin() -> None:
    workflow = _load(".github/workflows/e2e-required.yml")
    steps = workflow["jobs"]["e2e-required"]["steps"]
    primary = _get_step(steps, "Run critical e2e suite (attempt 1)")
    retry = _get_step(steps, "Run critical e2e suite (retry)")
    assert "-p pytest_asyncio.plugin" in primary["run"]
    assert "-p pytest_asyncio.plugin" in retry["run"]


def test_watchlists_extension_e2e_uses_playwright_chromium() -> None:
    workflow = _load(".github/workflows/ui-watchlists-extension-e2e.yml")
    job = workflow["jobs"]["watchlists-extension-e2e"]
    env = job["env"]
    steps = job["steps"]

    assert "TLDW_E2E_PLAYWRIGHT_CHANNEL" not in env
    assert env["TLDW_E2E_EXTENSION_HEADLESS"] == "0"
    assert env["TLDW_E2E_EXTENSION_TARGET_WAIT_MS"] == "5000"
    assert env["TLDW_E2E_EXTENSION_MINIMAL_LOCALES"] == "1"
    assert not any(step.get("name") == "Verify system Chrome" for step in steps)

    install_step = _get_step(steps, "Install Playwright Chromium")
    assert install_step["working-directory"] == "apps/extension"
    assert install_step["timeout-minutes"] == 10
    assert install_step["run"] == "../node_modules/.bin/playwright install --with-deps chromium"


def test_watchlists_extension_spec_uses_built_extension_launcher() -> None:
    text = Path("apps/extension/tests/e2e/watchlists.spec.ts").read_text(
        encoding="utf-8"
    )

    assert "launchWithBuiltExtensionOrSkip" in text
    assert "launchWithExtensionOrSkip" not in text
    assert "allowOffline: true" in text


def test_frontend_e2e_tiers_install_portaudio_before_backend_dependency_setup() -> None:
    workflow = _load(".github/workflows/frontend-e2e-tiers.yml")
    for job_name in ("critical", "features", "admin"):
        _assert_portaudio_installed_before_python_setup(
            ".github/workflows/frontend-e2e-tiers.yml",
            job_name,
        )
        install_step = _get_step(
            workflow["jobs"][job_name]["steps"],
            "Install FFmpeg and PortAudio (Linux)",
        )
        assert install_step["with"]["install-ffmpeg"] == "false"


def test_frontend_ux_gates_skip_ffmpeg_but_keep_portaudio() -> None:
    for job_name in ("onboarding-gate", "smoke-gate"):
        _assert_setup_skips_ffmpeg_but_keeps_portaudio(
            ".github/workflows/frontend-ux-gates.yml",
            job_name,
        )


def test_onboarding_docs_gate_skips_ffmpeg_but_keeps_portaudio() -> None:
    _assert_setup_skips_ffmpeg_but_keeps_portaudio(
        ".github/workflows/onboarding-docs-gate.yml",
        "onboarding-docs-gate",
    )


def test_e2e_smoke_skips_ffmpeg_but_keeps_portaudio() -> None:
    _assert_setup_skips_ffmpeg_but_keeps_portaudio(
        ".github/workflows/e2e-smoke.yml",
        "e2e-smoke",
        step_name="Install FFmpeg and PortAudio deps",
    )


def test_e2e_smoke_uses_minimal_test_dependencies() -> None:
    workflow = _load(".github/workflows/e2e-smoke.yml")
    install_step = _get_step(workflow["jobs"]["e2e-smoke"]["steps"], "Install dependencies")
    install_script = install_step["run"]
    install_commands = "\n".join(
        line for line in install_script.splitlines() if not line.strip().startswith("#")
    )

    assert "pip install -e .[dev]" not in install_commands
    assert "locust" not in install_commands
    for package in ("pytest", "pytest-asyncio", "pytest-xdist", "pytest-timeout"):
        assert package in install_commands


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


def test_character_chat_rate_limits_installs_portaudio_before_python_deps() -> None:
    workflow = _load(".github/workflows/ci.yml")
    steps = workflow["jobs"]["character-chat-rate-limits"]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    setup_step = _get_step(steps, "Setup Python and deps (uv)")

    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-ffmpeg"] == "false"
    assert install_step["with"]["install-portaudio"] == "true"
    assert steps.index(install_step) < steps.index(setup_step)


def test_character_chat_rate_limits_job_is_scoped_to_rate_limit_tests() -> None:
    workflow = _load(".github/workflows/ci.yml")
    steps = workflow["jobs"]["character-chat-rate-limits"]["steps"]
    legacy_run = _get_step(steps, "Run Character_Chat rate-limit tests with TEST_MODE=0")["run"]
    new_run = _get_step(steps, "Run Character_Chat_NEW rate-limit tests with TEST_MODE=0")["run"]

    assert "tldw_Server_API/tests/Character_Chat/test_rate_limits_specific.py" in legacy_run
    assert "tldw_Server_API/tests/Character_Chat -rs" not in legacy_run
    assert "-m rate_limit" in new_run
    assert "tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py" in new_run
    assert "tldw_Server_API/tests/Character_Chat_NEW -rs" not in new_run


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


def test_windows_research_shard_has_bounded_per_test_timeout() -> None:
    workflow = _load(".github/workflows/ci.yml")
    steps = workflow["jobs"]["full-suite-windows-312-shards"]["steps"]
    run_script = str(_get_step(steps, "Run OS shard tests")["run"])

    assert 'if [ "$SHARD_NAME" = "research-websearch" ]; then' in run_script
    assert (
        "EXTRA_PYTEST_ARGS=(-vv -p pytest_timeout --timeout=120 --timeout-method=thread)"
        in run_script
    )
    assert '"${EXTRA_PYTEST_ARGS[@]}"' in run_script


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
        assert "rag-new-" not in str(step.get("if"))


def test_embedding_model_setup_skips_non_embedding_chunking_shard() -> None:
    workflow = _load(".github/workflows/ci.yml")
    embedding_setup_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") in {"Cache embedding models", "Pre-download embedding models"}
    ]

    assert len(embedding_setup_steps) == 10
    for step in embedding_setup_steps:
        condition = str(step.get("if"))
        assert "ai-chunking-code-json-xml" not in condition
        assert "ai-chunking-templates" not in condition


def test_embedding_model_predownload_skips_backpressure_shard() -> None:
    workflow = _load(".github/workflows/ci.yml")
    predownload_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Pre-download embedding models"
    ]

    assert len(predownload_steps) == 5
    for step in predownload_steps:
        condition = str(step.get("if"))
        run_script = str(step.get("run") or "")
        assert "matrix.shard.name != 'ai-embeddings-backpressure'" in condition
        assert "matrix.shard.name != 'ai-embeddings-dlq-config'" in condition
        assert "matrix.shard.name != 'ai-embeddings-media-validation'" in condition
        assert "matrix.shard.name != 'ai-embeddings-policy'" in condition
        assert "env.RUN_MODEL_TESTS == '1'" in condition
        assert "env.RUN_REAL_HF_EMBEDDING_TESTS == '1'" in condition
        assert "env.RUN_REAL_HF_EMBEDDING_TESTS == 'true'" in condition
        assert "startsWith(matrix.shard.name, 'rag-new-')" not in condition
        assert '[[ "$SHARD_NAME" == "ai-embeddings-v5-core" ]]' not in run_script
        assert run_script.count("download_embedding_models.py") == 1
        assert "--skip-defaults --model sentence-transformers/all-MiniLM-L6-v2" in run_script
        assert (
            "python Helper_Scripts/download_embedding_models.py --target models/embeddings\n"
            not in run_script
        )


def test_setup_ffmpeg_action_can_skip_ffmpeg_but_keep_portaudio() -> None:
    action = _load(".github/actions/setup-ffmpeg/action.yml")
    assert action["inputs"]["install-ffmpeg"]["default"] == "true"

    linux_step = _get_step(action["runs"]["steps"], "Install FFmpeg (Linux)")
    assert (
        "runner.os == 'Linux' && (inputs.install-ffmpeg == 'true' || inputs.install-portaudio == 'true')"
        in linux_step["if"]
    )
    linux_script = linux_step["run"]
    assert 'inputs.install-ffmpeg' in linux_script
    assert (
        "grep -rl 'packages.microsoft.com' /etc/apt/sources.list.d | xargs -r sudo rm -f || true"
        in linux_script
    )
    assert (
        "grep -rl 'azure.archive.ubuntu.com' /etc/apt/sources.list.d | xargs -r sudo sed -i"
        in linux_script
    )
    assert (
        "grep -rl 'azure.archive.ubuntu.com' /etc/apt/sources.list.d | xargs -r sudo sed -i "
        "'s|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' || true"
        in linux_script
    )
    assert "Acquire::http::Timeout=20" in linux_script
    assert (
        "sudo apt-get install -y --no-install-recommends ffmpeg portaudio19-dev python3-all-dev"
        in linux_script
    )
    assert (
        "sudo apt-get install -y --no-install-recommends portaudio19-dev python3-all-dev"
        in linux_script
    )

    windows_step = _get_step(action["runs"]["steps"], "Install FFmpeg (Windows)")
    assert "inputs.install-ffmpeg == 'true'" in windows_step["if"]


def test_wait_for_postgres_action_bounds_linux_client_install() -> None:
    action = _load(".github/actions/wait-for-postgres/action.yml")
    install_step = _get_step(action["runs"]["steps"], "Install client (Linux)")
    install_script = install_step["run"]
    assert "command -v pg_isready" in install_script
    assert (
        "grep -rl 'azure.archive.ubuntu.com' /etc/apt/sources.list.d | xargs -r sudo sed -i "
        "'s|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' || true"
        in install_script
    )
    assert "Acquire::http::Timeout=20" in install_script
    assert "Acquire::https::Timeout=20" in install_script
    assert "sudo apt-get install -y --no-install-recommends postgresql-client" in install_script


def test_full_suite_ffmpeg_setup_scopes_heavy_install_to_media_runtime_shards() -> None:
    workflow = _load(".github/workflows/ci.yml")
    matrix_jobs = [
        "full-suite-linux-311-smoke",
        "full-suite-linux-312-shards",
        "full-suite-linux-313-shards",
        "full-suite-macos-312-shards",
        "full-suite-windows-312-shards",
        "full-suite-os-313-release-shards",
    ]
    expected_condition = (
        "${{ startsWith(matrix.shard.name, 'media-') || "
        "matrix.shard.name == 'ai-embeddings-media-validation' || "
        "matrix.shard.name == 'rag-new-unit-media-ingest' || "
        "matrix.shard.name == 'product-claims-service' || "
        "(matrix.shard.name == 'platform-services-core' || "
        "matrix.shard.name == 'platform-services-main-routing' || "
        "matrix.shard.name == 'platform-services-main-pollers') }}"
    )

    wizard_step = _get_step(workflow["jobs"]["wizard-tests"]["steps"], "Install FFmpeg and PortAudio (Linux)")
    assert wizard_step["with"]["install-ffmpeg"] == "true"

    for job_name in matrix_jobs:
        setup_steps = [
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("uses") == "./.github/actions/setup-ffmpeg"
        ]
        assert len(setup_steps) == 1
        setup_step = setup_steps[0]
        assert setup_step["with"]["install-portaudio"] == "true"
        assert setup_step["with"]["install-ffmpeg"] == expected_condition
        assert "product-watchlists" not in setup_step["with"]["install-ffmpeg"]
        assert "product-workflows" not in setup_step["with"]["install-ffmpeg"]


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


def test_full_suite_summaries_follow_backend_path_filter() -> None:
    workflow = _load(".github/workflows/ci.yml")
    expected_if = """always() && !cancelled() && (
  (
    github.event_name == 'workflow_run' &&
    needs.admission.result == 'success' &&
    needs.admission.outputs.should_run == 'true'
  ) ||
  github.event_name != 'workflow_run'
) && (
  (github.event_name != 'pull_request' &&
   github.event_name != 'workflow_run') ||
  needs.changes.outputs.backend_changed == 'true'
)"""
    summary_to_shards = {
        "full-suite-linux-312-summary": "full-suite-linux-312-shards",
        "full-suite-linux-313-summary": "full-suite-linux-313-shards",
        "full-suite-macos-312-summary": "full-suite-macos-312-shards",
        "full-suite-windows-312-summary": "full-suite-windows-312-shards",
    }

    for summary_job, shard_job in summary_to_shards.items():
        job = workflow["jobs"][summary_job]
        assert job["needs"] == [shard_job, "changes", "admission"]
        assert job["if"] == expected_if


def test_linux_311_smoke_is_sharded_for_timeout_control() -> None:
    workflow = _load(".github/workflows/ci.yml")
    job = workflow["jobs"]["full-suite-linux-311-smoke"]
    assert job["name"] == "Full Suite (Ubuntu / Python 3.11 / ${{ matrix.shard.name }})"

    shards = job["strategy"]["matrix"]["shard"]
    shard_paths = {shard["name"]: set(str(shard["paths"]).split()) for shard in shards}
    assert set(shard_paths) == {
        "authnz-unit",
        "config-core-loaders",
        "config-effective-api",
        "config-module-yaml",
        "config-routes-startup",
        "config-runtime-env",
        "core",
        "utils-http",
    }
    assert shard_paths["authnz-unit"] == {"tldw_Server_API/tests/AuthNZ_Unit"}
    assert shard_paths["utils-http"] == {
        "tldw_Server_API/tests/Utils",
        "tldw_Server_API/tests/http_client",
    }
    config_shards = {name for name in shard_paths if name.startswith("config-")}
    config_paths = [path for name in config_shards for path in shard_paths[name]]
    expected_config_paths = {
        str(path)
        for path in Path("tldw_Server_API/tests/Config").glob("test_*.py")
    }
    assert len(config_paths) == len(set(config_paths))
    assert set(config_paths) == expected_config_paths
    assert "tldw_Server_API/tests/Config" not in config_paths
    assert shard_paths["config-effective-api"] == {
        "tldw_Server_API/tests/Config/test_effective_config_api.py"
    }
    assert shard_paths["config-module-yaml"] == {
        "tldw_Server_API/tests/Config/test_module_yaml_integration.py"
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
        assert "chat-character-db" not in shard_names
        assert "product-claims" not in shard_names
        assert "product-evaluations" not in shard_names
        assert "product-prompts-workflows" not in shard_names
        assert "platform-mcp" not in shard_names
        assert "core-audit-security" not in shard_names
        assert {
            "core-audit-support",
            "core-audit-unified",
            "core-config",
            "core-security",
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
            "auth-core-root-property",
            "auth-core-unit-a-l",
            "auth-core-unit-m-z",
            "auth-integration-admin-auth",
            "auth-integration-authnz",
            "auth-integration-b-z",
            "auth-postgres",
            "auth-sqlite",
            "auth-unit-a-l",
            "auth-unit-m-z",
            "chacha-core-stores",
            "chacha-character-conversation",
            "chacha-content-persona",
            "db-privileges",
        }
        assert auth_db_shards.issubset(shard_names)
        assert "auth-core" not in shard_names
        assert {
            "ai-chromadb",
            "ai-chunking-code-json-xml",
            "ai-chunking-core",
            "ai-chunking-semantic-security",
            "ai-chunking-templates",
            "ai-embeddings-async",
            "ai-embeddings-backpressure",
            "ai-embeddings-chromadb-core",
            "ai-embeddings-core",
            "ai-embeddings-dlq-config",
            "ai-embeddings-hyde-ledger",
            "ai-embeddings-jobs-runtime",
            "ai-embeddings-media-validation",
            "ai-embeddings-observability",
            "ai-embeddings-policy",
            "ai-embeddings-v5-core",
            "ai-embeddings-v5-integration",
            "vector-stores-api",
            "vector-stores-integration",
            "vector-stores-unit",
            "paper-search",
            "rag-legacy",
            "research-websearch",
        }.issubset(shard_names)
        assert rag_new_shards.issubset(shard_names)
        assert {
            "chat-character-legacy-core",
            "chat-character-legacy-files",
            "chat-character-legacy-worldbook",
            "chat-character-db-core",
            "chat-character-db-api",
            "chat-character-unit-core",
            "chat-character-unit-chat",
            "chat-character-unit-persona",
            "chat-character-unit-prd",
            "chat-character-property",
            "chat-character-integration-api",
            "chat-character-integration-chat",
            "chat-character-integration-context",
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
        chat_core_shards = {
            "chat-legacy-integration",
            "chat-legacy-unit-a-l",
            "chat-legacy-unit-m-z",
            "chat-new-integration-property",
            "chat-new-unit-a-l",
            "chat-new-unit-m-z",
            "chatbooks-streaming",
        }
        assert chat_core_shards.issubset(shard_names)
        assert "chat-core" not in shard_names
        workflow_shards = {
            "product-workflows-adapters-core",
            "product-workflows-api",
            "product-workflows-engine",
            "product-workflows-runtime",
            "product-workflows-storage",
            "product-workflows-step-adapters",
            "product-workflows-step-capabilities",
            "product-workflows-step-registry",
        }
        product_shards = {
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
            "product-watchlists-a-r",
            "product-watchlists-core",
            "product-watchlists-pipeline",
        } | workflow_shards
        assert product_shards.issubset(shard_names)
        assert "product-workflows" not in shard_names
        assert "product-workflows-adapters" not in shard_names
        assert "product-watchlists" not in shard_names
        platform_shards = {
            "platform-infrastructure-metrics",
            "platform-mcp-core",
            "platform-resource-governance",
            "platform-sandbox-admin-artifacts",
            "platform-sandbox-runtimes",
            "platform-sandbox-state-store",
            "platform-sandbox-ws-streams",
            "platform-services-core",
            "platform-services-main-pollers",
            "platform-services-main-routing",
            "platform-services-shutdown-lifespan",
            "platform-services-startup",
        }
        assert platform_shards.issubset(shard_names)
        shard_paths = {shard["name"]: shard["paths"] for shard in shards}
        shard_path_sets = {
            name: set(str(paths).split()) for name, paths in shard_paths.items()
        }
        media_ingestion_shards = {
            "media-core-documents",
            "media-core-api",
            "media-ingestion-new-ocr",
            "media-ingestion-new-integration",
            "media-ingestion-new-unit-core",
            "media-ingestion-new-unit-mediawiki",
            "media-ingestion-new-unit-persistence",
            "media-ingestion-new-unit-processing",
            "media-ingestion-modification",
        }
        assert ({"media-audio", "media-legacy-free"} | media_ingestion_shards).issubset(
            shard_names
        )
        assert "media-ingestion" not in shard_names
        assert "media-ingestion-new-unit" not in shard_names
        assert shard_path_sets["media-audio"] == {
            "tldw_Server_API/tests/Audio",
            "tldw_Server_API/tests/AudioJobs",
            "tldw_Server_API/tests/Audio_Studio",
            "tldw_Server_API/tests/Benchmarks",
            "tldw_Server_API/tests/STT",
            "tldw_Server_API/tests/TTS",
            "tldw_Server_API/tests/TTS_NEW",
            "tldw_Server_API/tests/VLM",
        }
        assert shard_path_sets["media-core-documents"] == {
            "tldw_Server_API/tests/Media/test_document*.py",
            "tldw_Server_API/tests/Media/test_pdf_text_normalization.py",
        }
        assert shard_path_sets["media-core-api"] == {
            "tldw_Server_API/tests/Media/test_archive_member_cap.py",
            "tldw_Server_API/tests/Media/test_auto_chunking_process_endpoints.py",
            "tldw_Server_API/tests/Media/test_cache_index.py",
            "tldw_Server_API/tests/Media/test_ingest_web_content_endpoint_sanitization.py",
            "tldw_Server_API/tests/Media/test_json_*.py",
            "tldw_Server_API/tests/Media/test_media_*.py",
            "tldw_Server_API/tests/Media/test_navigation_policy_contract.py",
            "tldw_Server_API/tests/Media/test_process_code_and_uploads.py",
            "tldw_Server_API/tests/Media/test_upload_sink_security.py",
            "tldw_Server_API/tests/Media/unit",
        }
        assert shard_path_sets["media-ingestion-new-ocr"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/test_*.py"
        }
        assert shard_path_sets["media-ingestion-new-integration"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/integration"
        }
        assert shard_path_sets["media-ingestion-new-unit-core"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_archive_and_sanitization.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_auto_chunking_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_book_zip_safe_extract.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_book_processing_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_ebook_safe_paths.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_email_endpoint_error_mapping.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_file_validation.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_filename_and_mime_and_archive.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_ingest_jobs_batch_lookup.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_canonical_helpers.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_list_no_slash_redirect.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_upload_failures.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py",
        }
        assert shard_path_sets["media-ingestion-new-unit-mediawiki"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_ms_g_eval_validation.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_pdf_analysis_regressions.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_pdf_safe_paths.py",
        }
        assert shard_path_sets["media-ingestion-new-unit-persistence"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_*.py",
        }
        assert shard_path_sets["media-ingestion-new-unit-processing"] == {
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_document_like_item_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_transcription_models_endpoint.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_utils_time_conversion.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_video_*.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_visual_ingestion.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_xml_ingestion.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py",
            "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_youtube_audio_downloads.py",
        }
        assert shard_path_sets["media-ingestion-modification"] == {
            "tldw_Server_API/tests/Media_Ingestion_Modification"
        }
        assert shard_path_sets["media-legacy-free"] == {
            "tldw_Server_API/tests/Media",
            "tldw_Server_API/tests/MediaIngestion_NEW",
        }
        media_ingestion_test_files = sorted(
            str(path)
            for dirname in (
                "Media",
                "MediaIngestion_NEW",
                "Media_Ingestion_Modification",
            )
            for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
        )
        covered_media_files: dict[str, str] = {}
        for shard_name in media_ingestion_shards:
            for pattern in shard_path_sets[shard_name]:
                if Path(pattern).is_dir():
                    matches = [
                        filename
                        for filename in media_ingestion_test_files
                        if filename.startswith(f"{pattern}/")
                    ]
                else:
                    matches = sorted(fnmatch.filter(media_ingestion_test_files, pattern))
                assert matches, f"{shard_name} pattern {pattern} did not match any media tests"
                for filename in matches:
                    assert filename not in covered_media_files, (
                        f"{filename} covered by both {covered_media_files[filename]} and {shard_name}"
                    )
                    covered_media_files[filename] = shard_name
        assert set(covered_media_files) == set(media_ingestion_test_files)
        assert "tldw_Server_API/tests/VectorStores" not in shard_path_sets["ai-embeddings-core"]
        assert "vector-stores" not in shard_names
        assert shard_path_sets["ai-chromadb"] == {"tldw_Server_API/tests/ChromaDB"}
        assert "tldw_Server_API/tests/Admin" not in shard_path_sets["core-server-smoke"]
        assert "tldw_Server_API/tests/RAG_NEW" not in shard_path_sets["rag-legacy"]
        for shard_name in rag_new_shards:
            assert "tldw_Server_API/tests/RAG" not in shard_path_sets[shard_name]
        assert "tldw_Server_API/tests/PaperSearch" not in shard_path_sets["research-websearch"]
        assert "tldw_Server_API/tests/WebSearch" not in shard_path_sets["paper-search"]
        assert "tldw_Server_API/tests/Characters" not in shard_path_sets["chat-character-legacy-core"]
        assert "tldw_Server_API/tests/Characters" not in shard_path_sets["chat-character-db-core"]
        assert "tldw_Server_API/tests/Characters" not in shard_path_sets["chat-character-db-api"]
        assert "tldw_Server_API/tests/Character_Chat_NEW/unit" not in shard_path_sets["chat-character-property"]
        assert shard_path_sets["chat-legacy-integration"] == {
            "tldw_Server_API/tests/Chat/test*.py",
            "tldw_Server_API/tests/Chat/integration",
        }
        assert shard_path_sets["chat-legacy-unit-a-l"] == {
            "tldw_Server_API/tests/Chat/unit/test_[a-l]*.py"
        }
        assert shard_path_sets["chat-legacy-unit-m-z"] == {
            "tldw_Server_API/tests/Chat/unit/test_[m-z]*.py"
        }
        assert shard_path_sets["chat-new-integration-property"] == {
            "tldw_Server_API/tests/Chat_NEW/integration",
            "tldw_Server_API/tests/Chat_NEW/property",
        }
        assert shard_path_sets["chat-new-unit-a-l"] == {
            "tldw_Server_API/tests/Chat_NEW/unit/test_[a-l]*.py"
        }
        assert shard_path_sets["chat-new-unit-m-z"] == {
            "tldw_Server_API/tests/Chat_NEW/unit/test_[m-z]*.py"
        }
        assert shard_path_sets["chatbooks-streaming"] == {
            "tldw_Server_API/tests/Chatbooks",
            "tldw_Server_API/tests/Explainer",
            "tldw_Server_API/tests/Streaming",
        }
        chat_core_files = {
            str(path)
            for dirname in ("Chat", "Chat_NEW", "Chatbooks", "Explainer", "Streaming")
            for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
        }
        covered_chat_core_files: dict[str, str] = {}
        for shard_name in chat_core_shards:
            for pattern in shard_path_sets[shard_name]:
                if Path(pattern).is_dir():
                    prefix = f"{pattern.rstrip('/')}/"
                    matches = {
                        filename
                        for filename in chat_core_files
                        if filename.startswith(prefix)
                    }
                else:
                    matches = {
                        filename
                        for filename in chat_core_files
                        if fnmatch.fnmatch(filename, pattern)
                    }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_chat_core_files, (
                        f"{filename} matched both "
                        f"{covered_chat_core_files[filename]} and {shard_name}"
                    )
                    covered_chat_core_files[filename] = shard_name
        assert set(covered_chat_core_files) == chat_core_files
        assert "tldw_Server_API/tests/Claims" not in shard_path_sets["product-collections"]
        assert "tldw_Server_API/tests/Evaluations" not in shard_path_sets["product-claims-core"]
        watchlist_shards = {
            "product-watchlists-a-r",
            "product-watchlists-core",
            "product-watchlists-pipeline",
        }
        watchlist_test_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Watchlists").glob("test*.py")
        }
        covered_watchlist_files: dict[str, str] = {}
        for shard_name in watchlist_shards:
            for pattern in shard_path_sets[shard_name]:
                matches = sorted(fnmatch.filter(watchlist_test_files, pattern))
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    if filename in covered_watchlist_files:
                        raise AssertionError(
                            f"{filename} is listed by multiple watchlist shards: "
                            f"{covered_watchlist_files[filename]} and {shard_name}"
                        )
                    covered_watchlist_files[filename] = shard_name
        assert set(covered_watchlist_files) == watchlist_test_files

        core_shards = {
            "core-audit-support",
            "core-audit-unified",
            "core-config",
            "core-security",
            "core-server-smoke",
            "core-setup-usage",
            "core-utils-tooling",
        }
        core_dirs = (
            "Audit",
            "Config",
            "Context_Integrity",
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
        # Individual files outside auth_db_dirs that are legitimately bundled
        # into an auth/db shard (e.g. persona-adjacent API tests living under
        # a feature directory rather than a DB-specific one).
        auth_db_extra_files = {
            "tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_artifact_validation.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle_postgres.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_context_api.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_job_status.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py",
            "tldw_Server_API/tests/Workspaces/test_workspace_source_saved_views_api.py",
        }
        auth_db_files = {
            str(path)
            for dirname in auth_db_dirs
            for path in Path("tldw_Server_API/tests", dirname).glob("**/test*.py")
        } | auth_db_extra_files
        covered_auth_db_files: dict[str, str] = {}
        for shard_name in auth_db_shards:
            for pattern in shard_path_sets[shard_name]:
                assert (
                    any(
                        pattern == root.rstrip("/") or pattern.startswith(root)
                        for root in auth_db_roots
                    )
                    or pattern in auth_db_extra_files
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
            "ai-embeddings-async",
            "ai-embeddings-backpressure",
            "ai-embeddings-chromadb-core",
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

        vector_store_shards = {
            "vector-stores-api",
            "vector-stores-integration",
            "vector-stores-unit",
        }
        vector_store_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/VectorStores").glob("**/test*.py")
        }
        covered_vector_store_files: dict[str, str] = {}
        for shard_name in vector_store_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/VectorStores/")
                if Path(pattern).is_dir():
                    prefix = f"{pattern.rstrip('/')}/"
                    matches = {
                        filename
                        for filename in vector_store_files
                        if filename.startswith(prefix)
                    }
                else:
                    matches = {
                        filename
                        for filename in vector_store_files
                        if fnmatch.fnmatch(filename, pattern)
                    }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_vector_store_files, (
                        f"{filename} matched both "
                        f"{covered_vector_store_files[filename]} and {shard_name}"
                    )
                    covered_vector_store_files[filename] = shard_name

        assert set(covered_vector_store_files) == vector_store_files

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
        visual_identity_character_file = (
            "tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py"
        )
        assert shard_path_sets["visual-identities"] == {
            visual_identity_character_file,
            "tldw_Server_API/tests/Visual_Identities",
        }
        # test_character_rate_limiter_429.py lives outside Character_Chat but
        # is bundled with the legacy character rate-limit tests it mirrors.
        legacy_character_extra_files = {
            "tldw_Server_API/tests/RateLimiting/test_character_rate_limiter_429.py",
        }
        legacy_character_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Character_Chat").glob("**/test*.py")
        } | legacy_character_extra_files
        covered_legacy_character_files: dict[str, str] = {}
        for shard_name in legacy_character_shards | {"visual-identities"}:
            for pattern in shard_path_sets[shard_name]:
                if shard_name == "visual-identities" and pattern != visual_identity_character_file:
                    continue
                assert (
                    pattern.startswith("tldw_Server_API/tests/Character_Chat/")
                    or pattern in legacy_character_extra_files
                )
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

        character_db_shards = {
            "chat-character-db-core",
            "chat-character-db-api",
        }
        character_db_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Characters").glob("test*.py")
        }
        covered_character_db_files: dict[str, str] = {}
        for shard_name in character_db_shards:
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Characters/")
                matches = {
                    filename
                    for filename in character_db_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_character_db_files, (
                        f"{filename} matched both "
                        f"{covered_character_db_files[filename]} and {shard_name}"
                    )
                    covered_character_db_files[filename] = shard_name

        assert set(covered_character_db_files) == character_db_files

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
            for path in Path("tldw_Server_API/tests/Claims").glob("**/test*.py")
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

        workflow_files = {
            str(path)
            for path in Path("tldw_Server_API/tests/Workflows").glob("**/test*.py")
        }
        covered_workflow_files: dict[str, str] = {}
        for shard_name in workflow_shards:
            assert "tldw_Server_API/tests/Workflows" not in shard_path_sets[shard_name]
            for pattern in shard_path_sets[shard_name]:
                assert pattern.startswith("tldw_Server_API/tests/Workflows/")
                matches = {
                    filename
                    for filename in workflow_files
                    if fnmatch.fnmatch(filename, pattern)
                }
                assert matches, f"{shard_name} pattern matched no files: {pattern}"
                for filename in matches:
                    assert filename not in covered_workflow_files, (
                        f"{filename} matched both "
                        f"{covered_workflow_files[filename]} and {shard_name}"
                    )
                    covered_workflow_files[filename] = shard_name

        assert set(covered_workflow_files) == workflow_files


def test_legacy_free_media_checks_run_on_dedicated_shard() -> None:
    workflow = _load(".github/workflows/ci.yml")
    legacy_free_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Run legacy-free media checks"
    ]

    assert len(legacy_free_steps) == 5
    for step in legacy_free_steps:
        assert step.get("if") == "matrix.shard.name == 'media-legacy-free'"
        assert 'mkdir -p "$RESULTS_DIR"' in step["run"]


def test_regular_full_suite_steps_skip_dedicated_media_legacy_free_shard() -> None:
    workflow = _load(".github/workflows/ci.yml")
    regular_shard_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name")
        in {
            "Run shard tests",
            "Run OS shard tests",
            "Run release OS shard tests",
        }
    ]

    assert len(regular_shard_steps) == 5
    for step in regular_shard_steps:
        assert step.get("if") == "matrix.shard.name != 'media-legacy-free'"
