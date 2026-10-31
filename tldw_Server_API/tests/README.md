# Test Suite Guide

This directory contains the end-to-end, integration, and unit tests for `tldw_server`. The notes below summarize how to run the suite and the knobs that control which tests execute.

## Quick Start
- Install dependencies with `pip install -e .[dev]`.
- Install PG drivers: pip install -e .[db_postgres]
  - Set DSN: `export PG_TEST_DSN=postgresql://tldw_user:TestPassword123!@localhost:5432/tldw_content`
- Run the whole suite: `python -m pytest tldw_Server_API/tests -q`.
- Run by marker: `python -m pytest -m "unit"` or `pytest -m "integration and not slow"`.
- Use `pytest --maxfail=1 -x` while iterating to stop on the first failure.

## Where Does a New Test Go?
Several modules have more than one test tree. The `_NEW` suffix does **not** mean "newer" or
"preferred", and no tree here is legacy. Conftest fixtures are scoped to their directory, so the
tree you pick decides which fixtures and env presets your test gets. Do not start new `_NEW` or
per-module sibling trees; use markers (`unit`, `integration`, ...) and subdirectories instead.

| Module | Put the test in | When |
|---|---|---|
| Chat | `Chat/unit`, `Chat/integration` (default) | Chat service, provider routing, orchestration, ChaCha deps, endpoint integration. Uses the shared `_plugins.chat_fixtures`. |
| | `Chat_NEW/{unit,integration,property}` | `/chat/completions` public contract tests that want its conftest presets (`TEST_MODE`, `DEFAULT_LLM_PROVIDER=openai`, bearer key), and property tests. |
| RAG | `RAG_NEW/{unit,integration,property}` (default) | Pipeline, retrieval, reranking, caching, endpoints. Its autouse fixtures isolate the semantic cache and provider overrides. |
| | `RAG/` | Knowledge QA, feedback, analytics, and dual-backend regressions. |
| | either | Anything that runs real SQL must use `dual_backend_env` (SQLite + Postgres). It is defined in `RAG/conftest.py` and re-exported by `RAG_NEW/conftest.py`, so it works in both trees. |
| TTS | `TTS/` | Security, sanitization, and adapter internals. |
| | `TTS_NEW/{unit,integration,property}` | Public contract: endpoints, adapter registry behaviour, provider switching. The two trees split coverage between them and don't overlap, so don't copy a test into both. |
| AuthNZ | `AuthNZ_Unit/` | Pure unit tests with no database. |
| | `AuthNZ_SQLite/`, `AuthNZ_Postgres/` | Tests that only make sense on that backend. `AuthNZ_Postgres` always gets a Postgres DSN. |
| | `AuthNZ_Federation/` | Federation/SSO. |
| | `AuthNZ/` | Everything else (API, integration, property). |
| Ingestion / Media | the tree that already tests the same source module | The ingestion trees have no single axis. Find where the module you changed is imported (`grep -rl <module> tldw_Server_API/tests`) and add to that tree. |

CI shard names in `.github/workflows/ci.yml` are chosen to balance wall-clock time. They do not
describe the domain. For example, AuthNZ is sliced alphabetically across shards, and
`chat-*` / `chat-new-*` are the `Chat` / `Chat_NEW` trees.

## Pytest Markers & CLI Switches
- Common markers registered across the suite: `unit`, `integration`, `slow`, `stress`, `external_api`, `pg_jobs`, `requires_model`, `multi_user`, `single_user`.
- View the full list any time with `pytest --markers`.
- ChromaDB tests accept `--run-model-tests` (or `RUN_MODEL_TESTS=1`) to pull lightweight embedding models.
- E2E tests add `--skip-slow`, `--critical-only`, and `--auth-mode={auto|single_user|multi_user}` (see `tests/e2e/README.md` for the full flow).

## Environment Toggles

### Database & Auth
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `RUN_PG_INTEGRATION` | disabled | Enables Postgres-backed AuthNZ integration tests such as `AuthNZ/test_media_permission_enforcement.py`. | Requires `DATABASE_URL`/`TEST_DATABASE_URL` pointing at PostgreSQL and credentials (`TEST_DB_*`). |
| `TLDW_TEST_POSTGRES_REQUIRED` | disabled | Fail fast instead of skipping when Postgres cannot be reached. | Used by AuthNZ and Prompt Studio heavy suites; combine with the Postgres variables below. |
| `TEST_DATABASE_URL` / `DATABASE_URL` | varies | Provides a full Postgres DSN for AuthNZ fixtures. | Overrides `TEST_DB_HOST`, `TEST_DB_PORT`, `TEST_DB_NAME`, `TEST_DB_USER`, `TEST_DB_PASSWORD`. |
| `TLDW_TEST_NO_DOCKER` | disabled | Prevents AuthNZ fixtures from auto-starting a local Postgres Docker container. | Set when you manage Postgres yourself. |
| `TLDW_TEST_PG_IMAGE` | `postgres:18` | Selects the Docker image used when the fixtures auto-start Postgres. | Combine with `TLDW_TEST_PG_CONTAINER_NAME` if needed. |
| `RUN_PG_JOBS_STRESS` | disabled | Enables `tests/Jobs/test_jobs_pg_concurrency_stress.py` (heavy multi-process stress). | Requires `JOBS_DB_URL`/`POSTGRES_TEST_DSN`. |
| `RUN_PG_JOBS_STRESS_STRICT` | disabled | Extends the PG jobs stress test with stricter coverage assertions. | Only meaningful when `RUN_PG_JOBS_STRESS=1`. |

### Prompt Studio & Research
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `TLDW_PS_BACKEND` | `sqlite` | Chooses the backend (`sqlite` or `postgres`) for the heavy Prompt Studio dual-backend suite. | Located in `prompt_studio/integration/test_optimizations_dual_backend_heavy.py`. |
| `TLDW_PS_STRESS` | disabled | Expands Prompt Studio heavy tests with larger corpora and more iterations. | Optional tuning knobs: `TLDW_PS_TC_COUNT`, `TLDW_PS_ITERATIONS`, `TLDW_PS_OPT_COUNT`. |
| `RUN_EXTERNAL_API_TESTS` | disabled | Enables PaperSearch tests that hit live third-party APIs. | Provide provider keys such as `SPRINGER_NATURE_API_KEY`, `IEEE_API_KEY`, `ELSEVIER_API_KEY` before enabling. |

### LLM / Embeddings / Chat
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `RUN_MODEL_TESTS` or `--run-model-tests` | disabled | Allows ChromaDB tests marked `requires_model` to download tiny HuggingFace models. | Requires network access. |
| `RUN_REAL_EMBEDDINGS` | disabled | Allows embeddings integration tests to download and execute real HuggingFace models in CI. | Combine with `HF_TOKEN` if the model requires authentication. |
| `RUN_STRESS_TESTS` | disabled | Executes the concurrent load test inside `test_embeddings_v5_integration.py`. | Heavy; opt-in only. |
| `RUN_COMMERCIAL_CHAT_TESTS` | disabled | Exercises live commercial chat providers (OpenAI, Anthropic, etc.). | Ensure the related API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, etc.) are populated with real credentials. |

### Audio, TTS, and Media
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `RUN_TTS_LEGACY_INTEGRATION` | disabled | Runs legacy TTS adapter tests under `tests/TTS/`. | Requires provider-specific keys (e.g., `ELEVENLABS_API_KEY`, local model paths). |
| `ECHO_TTS_MODULE_PATH` | unset | Points to an Echo-TTS checkout for the CUDA initialization test under `tests/TTS_NEW/integration/`. | If unset, the test looks for `../echo-tts` next to the repo. |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | unset | Force offline mode for the Echo-TTS CUDA generation test under `tests/TTS_NEW/integration/`. | The test only runs when the Echo-TTS model repos are already cached locally. |
| `ALLOW_HEAVY_AUDIO_SMOKE` | disabled | Enables the audio worker smoke test (`tests/AudioJobs/test_audio_worker_smoke.py`). | Install `ffmpeg` and ensure audio models are available. |
| `RUN_AUDIO_E2E` | disabled | Allows the optional audio upload E2E test. | Requires a running server plus audio processing dependencies. |

### Workflows, MCP, and Services
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `TLDW_WORKFLOW_STRESS` | disabled | Enables `tests/Workflows/test_workflow_stress.py` which hammers the workflow engine. | Intended for soak testing only. |
| `RUN_MCP_TESTS` | disabled | Activates MCP metrics endpoint tests. | Provide `SINGLE_USER_API_KEY` so the tests can authenticate. |

### End-to-End Harness
| Setting | Default | Effect | Notes |
| --- | --- | --- | --- |
| `E2E_TEST_BASE_URL` | `http://localhost:8000` | Base URL for API E2E tests. | Ensure the server is running before starting the suite. |
| `E2E_AUTH_MODE` | `auto` | Overrides auth mode detection for E2E tests. | Choose `single_user` or `multi_user` to force a mode. |
| `E2E_RATE_LIMIT_DELAY` | `0.5` | Seconds to wait before retrying on rate-limit responses. | Adjust if the server is under heavy load. |
| `E2E_MAX_RETRIES` | `3` | Retry attempts after rate limits. | Increase for slower deployments. |
| `E2E_SERVER_STARTUP_TIMEOUT` | `30` | Seconds to wait for a server to boot before declaring a failure. | Helpful when bootstrapping against Docker. |
| `E2E_ADMIN_BEARER` | unset | Bearer token used for admin-level E2E scenarios. | Required for multi-user onboarding tests. |

### Admin UI Playwright Checks
The frontend E2E tests for the Next.js admin UI live under `apps/tldw-frontend/e2e`.

- Required: the API server must be running in `single_user` mode with a valid `SINGLE_USER_API_KEY`.
- Optional: the test can auto-start the admin UI with `npm run dev` if `ADMIN_UI_URL` is not reachable.
  - Set `ADMIN_UI_URL` (default `http://127.0.0.1:3001`) if you run the admin UI on a different port.
  - Set `ADMIN_UI_API_KEY` if you want to override the key resolution.
  - If unset, the test will look in `.env`, `.env.local`, `tldw_Server_API/Config_Files/.env`,
    and `admin-ui/.env.local` / `admin-ui/.env` for `SINGLE_USER_API_KEY` or `NEXT_PUBLIC_X_API_KEY`.
  - The login flow supports `?mode=apikey` to force the API key form (useful when the UI is already running).

### Audio / LLM Provider Keys
Many suites look for provider-specific credentials when the corresponding feature flags are set. Typical examples include:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN`
- `ELEVENLABS_API_KEY`, `EXTERNAL_TRANSCRIPTION_TEST_API_KEY`
- `SPRINGER_NATURE_API_KEY`, `IEEE_API_KEY`, `ELSEVIER_API_KEY`

When a key is missing, the individual test will usually skip with a descriptive message.

## Tips
- Export environment variables in your shell or create a temporary `.env` when running targeted suites.
- Many integration fixtures automatically set `TEST_MODE=true` and disable rate limiting; you can do the same when writing new tests.
- Character_Chat rate-limit tests: run with `TEST_MODE=0` to exercise real limits. Example:
  - `TEST_MODE=0 pytest -q tldw_Server_API/tests/Character_Chat -k "rate_limit or max_messages or complete" -rs`
  - CI includes a dedicated job that runs the Character_Chat suite with `TEST_MODE=0` to catch regressions.
- Keep heavy toggles disabled by default in CI to control runtime. Enable them locally when validating provider integrations or stress scenarios.
### Postgres DSN for Tests
- Preferred: export a full DSN and all tests will use it:
  - `export TEST_DATABASE_URL=postgresql://USER:PASS@HOST:PORT/DB`
  - Example for the default dev container: `postgresql://tldw_user:TestPassword123!@127.0.0.1:5432/tldw_content`
- If no DSN is set, tests fall back in this order when building connection params:
  1) Container-style `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` (+ `POSTGRES_TEST_HOST`/`POSTGRES_TEST_PORT` if set)
  2) `TEST_DB_HOST`, `TEST_DB_PORT`, `TEST_DB_NAME`, `TEST_DB_USER`, `TEST_DB_PASSWORD`
  3) Project defaults (`127.0.0.1:5432`, `tldw_test`, `tldw_user`, `TestPassword123!`)
- A small helper `tests/helpers/pg_env.py` centralizes this logic. Import `get_pg_env()` or `pg_dsn()` in tests to avoid drift.

## Unified Postgres Fixtures (Auto-Provisioning)

The suite provides unified Postgres fixtures under `tests/_plugins/postgres.py`:

- `pg_server` (session): resolves connection parameters via `tests/helpers/pg_env.py` and ensures reachability. If the target is localhost and not reachable, it will attempt to auto-start a Docker Postgres (`postgres:18`).
- `pg_temp_db` (function): creates a fresh temporary database for each test via CREATE DATABASE and drops it at teardown. Returns a dict with `host`, `port`, `user`, `password`, `database`, `dsn`.
- `pg_eval_params` and `pg_database_config`: convenience wrappers used by various tests.

Automatic Docker fallback
- If the resolved host is local and either:
  - no Postgres is listening on the target port, or
  - CREATE DATABASE fails due to insufficient privileges on an existing server,
  the fixture will attempt to start a local Docker container with the resolved `user/password` and retry. If the primary port is busy but privileges are wrong, the fixture falls back to an alternate port (default `5434`).

Environment toggles
- `POSTGRES_TEST_DSN` / `JOBS_DB_URL` / `TEST_DATABASE_URL` / `DATABASE_URL`: explicit DSNs (highest precedence for different suites).
- Container-style parts: `POSTGRES_TEST_HOST`, `POSTGRES_TEST_PORT`, `POSTGRES_TEST_USER`, `POSTGRES_TEST_PASSWORD`, `POSTGRES_TEST_DB`.
- `TLDW_TEST_NO_DOCKER=1`: disable auto-start; tests will skip (or fail if required) when Postgres is unavailable.
- `TLDW_TEST_PG_IMAGE` (default `postgres:18`), `TLDW_TEST_PG_CONTAINER_NAME` (default `tldw_postgres_test`): control the auto-started container.
- `TLDW_TEST_PG_ALT_PORT` (default `5434`): alternate port used when an existing local Postgres is reachable but lacks privileges.
- `TLDW_TEST_PG_DEBUG=1`: verbose diagnostics from the fixture (resolved DSN, connect errors, fallback attempts).
- `TLDW_TEST_POSTGRES_REQUIRED=1`: fail instead of skip when Postgres cannot be provisioned.

Minimal commands
- Install drivers: `pip install -e .[db_postgres]`
- Let the fixtures provision Postgres automatically: `pytest -q -k "postgres" -rs`
- Show provisioning details: `TLDW_TEST_PG_DEBUG=1 pytest -q -k "postgres" -rs -s`

Jobs suite notes
- Jobs tests are gated off by default. Enable them with `RUN_JOBS=1`.
- Jobs PG tests use `jobs_pg_dsn` (which internally allocates a per-test DB via the unified fixtures). Example:
  - `RUN_JOBS=1 pytest -q tldw_Server_API/tests/Jobs -k "postgres and ttl" -rs`
