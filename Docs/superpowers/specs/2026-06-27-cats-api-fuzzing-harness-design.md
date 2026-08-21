# CATS API Fuzzing Harness Design

Backlog task: TASK-2369

## Context

tldw_server exposes a large FastAPI OpenAPI surface at `/openapi.json`. A
minimal-test OpenAPI generation pass produced 1608 paths and 2044 operations,
which is too broad for a single first-pass fuzzing run. CATS 13.8.0 is already
available locally and can run OpenAPI-driven negative and security fuzzing
against a live API server.

The harness should make broad coverage practical by splitting fuzzing into
explicit blocks. Each block should state its risk level, required seed state,
mutation behavior, network policy, timeout profile, and skip rationale. The
initial gate should find server crashes and unsafe runtime behavior without
turning expected validation, authentication, and missing-resource responses
into noise.

## Goals

- Provide a repeatable local and CI-friendly CATS fuzzing harness for the
  tldw_server API.
- Cover the API broadly through staged blocks instead of one monolithic sweep.
- Fail on unexpected server errors while preserving detailed reports for
  warnings and schema mismatches.
- Keep local developer data, provider credentials, and external networks out of
  default fuzzing runs.
- Make stateful coverage possible through seed data and CATS reference data.

## Non-Goals

- Replace pytest integration tests or property tests.
- Require real LLM, embedding, search, webhook, social, or provider services.
- Exercise all streaming, WebSocket, media, and background-job behavior in the
  first implementation slice.
- Make full broad fuzzing a required PR gate.

## Block Architecture

The harness should use a block manifest rather than hard-coded one-off CATS
commands. Each block entry should include:

- `name`: stable block id used in reports and CLI selection.
- `description`: short purpose and coverage summary.
- `risk`: one of `contract`, `public-read`, `auth-read`, `isolated-mutation`,
  `external-risk`, or `manual`.
- `paths` or `tags`: allowlist passed to CATS.
- `skip_paths`, `skip_tags`, or `skip_methods`: explicit exclusions.
- `requires_seed`: whether a seed phase must run before CATS.
- `allows_mutation`: whether the block may create, update, or delete data.
- `allows_network`: false by default.
- `timeout_seconds`, `read_timeout`, and `max_requests_per_minute`.
- `expected_gate`: default `no_5xx`, with stricter gates added later.
- `skip_reason`: required for risky paths excluded from broad blocks.

### Initial Blocks

1. `contract`
   - Runs `cats validate` and `cats stats` against a generated OpenAPI file.
   - Does not call the API service.
   - Records known OpenAPI validation defects separately from runtime failures.

2. `public-read`
   - Covers `/`, `/health`, `/ready`, `/health/ready`, `/api/v1/health/*`,
     `/api/v1/config/docs-info`, `/api/v1/config/quickstart`, and similar
     public metadata endpoints.
   - Uses blackbox mode and fails on `5xx`.
   - Runs in PR CI once stable.

3. `auth-read`
   - Covers authenticated read-only endpoints such as LLM provider lists,
     capabilities, metrics JSON, MCP status/catalog lists, RAG health, user
     profile read paths, and resource-governor diagnostics.
   - Uses `X-API-KEY` in single-user mode.
   - Scrubs all provider credentials from the environment.

4. `auth-crud-isolated`
   - Covers selected create/update/delete flows in temporary SQLite user data.
   - Requires seeding and CATS `refData` or `urlParams` so IDs reference real
     rows rather than mostly hitting 404 responses.
   - Runs nightly or manually first.

5. `media-light`
   - Covers upload and media-adjacent endpoints only with tiny local fixtures.
   - Skips remote downloads, transcription, large files, model execution, and
     provider calls.
   - Requires strict file-size and timeout controls.

6. `protocol-streaming`
   - Covers protocol and streaming-adjacent HTTP endpoints only where CATS can
     complete requests reliably.
   - Skips WebSocket-only behavior and long-lived SSE streams unless the block
     has short read timeouts and explicit path exclusions.

7. `broad-nightly`
   - Expands across larger tags/path groups in blackbox mode.
   - Excludes external-egress, destructive admin, long-running media, and
     provider-backed routes by default.
   - Intended for scheduled CI or local manual use, not mandatory PR gating.

8. `multi-user-auth`
   - Future block for JWT and RBAC coverage.
   - Seeds users, roles, and admin/non-admin JWTs.
   - Exercises invalid, missing, and insufficient-scope auth separately from
     single-user API key fuzzing.

## Runtime Isolation

The runner should start from an explicit test environment:

- `AUTH_MODE=single_user`.
- A deterministic long `SINGLE_USER_API_KEY` and `SINGLE_USER_TEST_API_KEY`.
- Temporary `DATABASE_URL`.
- Temporary `USER_DB_BASE_DIR`.
- Temporary output, upload, artifact, and report directories where supported.
- Disabled schedulers, background workers, webhook delivery, and external
  provider calls where supported by existing configuration flags.
- Low request rate and bounded read/write/connect timeouts.

The default runner should fail fast if it detects real provider credentials or
webhook destinations in the environment. Examples include `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, provider-specific API keys, Slack/Discord/Telegram tokens,
and webhook URLs. A deliberate override flag can be added later for controlled
external-provider fuzzing, but the default behavior must be local-only.

Credential scrubbing must account for the app configuration loader. The server
loads `TLDW_ENV_FILE` and canonical repo `.env` files with `override=False`, so
unsetting a key can allow a real value from `.env` to be loaded later. The
runner should build an explicit child process environment that pre-sets known
provider, webhook, and social integration variables to empty or inert sentinel
values, points `TLDW_ENV_FILE` at a temporary minimal fuzzing env file, and
verifies the effective child environment before starting uvicorn.

## CATS Invocation Shape

Each block should resolve to a deterministic CATS command similar to:

```bash
cats \
  -c <generated-openapi.json> \
  -s http://127.0.0.1:<port> \
  -H X-API-KEY=<test-key> \
  --maskHeaders X-API-KEY,Authorization \
  --blackbox \
  --skipReportingForIgnored \
  --maxRequestsPerMinute <block-limit> \
  --connectionTimeout <seconds> \
  --readTimeout <seconds> \
  --writeTimeout <seconds> \
  --reportFormat HTML_ONLY,JUNIT \
  --output <artifact-dir>/<block-name> \
  --path <comma-separated-path-allowlist>
```

The compact JSON summary is owned by the runner. CATS 13.8.0 does not expose a
JSON report format; supported report formats include `HTML_ONLY`, `HTML_JS`,
`HTML_JS_CLUSTERS`, and `JUNIT`.

The first implementation should prefer path allowlists because the current tag
set is very large and tag cohesion is uneven. Tag-based blocks can be added
after individual tags prove stable.

## Stateful Seed Data

Stateful blocks should have an explicit seed phase. The seed phase should:

- Run against the same isolated server and temp DB as the fuzz block.
- Create minimal valid entities for selected route families.
- Emit CATS `refData`, `urlParams`, and optional header/query files.
- Avoid real networks and large media.
- Record all generated IDs in a manifest for replay and cleanup.

Without seeding, stateful fuzzing would mostly test missing-resource paths and
would not meaningfully cover authorization, validation, or update logic.

## Failure Policy

Initial runtime blocks use blackbox mode. The gate is:

- fail on any CATS `error` caused by an HTTP `5xx`;
- report but do not initially fail on warnings, undocumented response codes,
  content type mismatches, or response schema mismatches;
- record expected `4xx` validation/auth failures as non-failures;
- keep full CATS artifacts for replay and manual triage.

Later hardening blocks should add stricter expectations:

- Missing or invalid auth should return `401` or `403`, not `2xx` or `5xx`.
- Public endpoints should stay public only when intentionally listed.
- Response schema and content type warnings should become actionable after the
  OpenAPI contract is stable enough for that block.

## Reporting

Every run should write a compact summary JSON next to the CATS reports. The
summary should include:

- CATS version.
- git commit and dirty-state marker when available.
- OpenAPI spec hash.
- block name and block manifest hash.
- server URL and port.
- paths, tags, and methods covered.
- skipped paths and reasons.
- request count, warnings, errors, and `5xx` count.
- random seed and replay references where CATS provides them.
- CATS exit code and failure class, separating API failures from CATS/tooling
  failures.
- masked command arguments plus stdout and stderr artifact paths.

Failure triage should group results by `method + path + status + fuzzer`, not
only by raw request count. This keeps broad blocks readable when many payloads
hit the same underlying issue.

## CI And Scheduling

PR CI after OpenAPI validation is clean:

- `contract`
- `public-read`

Until the known vector store OpenAPI examples issue is fixed, `contract` should
run only as an advisory local/manual check or use an explicit known-issues file
that keeps the non-blocking failure visible.

Nightly CI:

- `auth-read`
- selected `auth-crud-isolated`
- selected `media-light`

Manual or scheduled security sweeps:

- `protocol-streaming`
- `broad-nightly`
- future `multi-user-auth`

Nightly and manual runs should upload CATS HTML/JUnit reports plus the
runner-owned summary JSON as artifacts. PR CI should keep reports small and
avoid starting provider-backed or long-running workers.

## Known Issues

CATS currently reports the generated OpenAPI document as invalid because
`GET /api/v1/vector_stores/{store_id}/vectors` exposes three query parameter
`examples` fields in a shape CATS rejects:

- `filter`
- `order_by`
- `order_dir`

This is defined in `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py`.
The first implementation should either fix those examples before enabling the
`contract` gate or record the issue in a known-issues file so the gate remains
transparent. The preferred path is to fix the OpenAPI examples as a small
pre-harness cleanup task if the file is not owned by another active change.
Until one of those paths is complete, `contract` must not be a blocking PR gate.

## Security And Safety

Default fuzzing must be local-only. The runner should:

- pre-set sensitive environment variables to empty or inert sentinel values
  before starting the server so dotenv fallback files cannot reintroduce real
  credentials;
- set `TLDW_ENV_FILE` to a temporary minimal fuzzing env file;
- verify the effective child process environment before importing or starting
  the app;
- reject runs with real credentials unless explicitly overridden;
- default `allows_network` to false;
- skip web scraping, research provider, social integration, webhook, and LLM
  provider execution blocks unless deliberately selected;
- mask auth headers in console and reports;
- keep generated test keys long enough to satisfy AuthNZ validation;
- write all mutable data under temporary directories.

## Rollout Plan

1. Fix or explicitly known-issue the vector store OpenAPI examples, then
   implement the block manifest and runner with `contract` and `public-read`.
2. Add isolated authenticated server startup and `auth-read`.
3. Add seed helpers and `auth-crud-isolated` for one or two stable route
   families.
4. Add `media-light` with tiny fixtures and strict timeout controls.
5. Add `protocol-streaming` exclusions and short-read checks.
6. Add `broad-nightly` as a scheduled/manual artifact-producing run.
7. Add `multi-user-auth` after single-user API key fuzzing is stable.

## Verification Strategy

The design and implementation should be verified with:

- `cats --version`.
- `cats validate -c <generated-openapi.json>`.
- `cats stats -c <generated-openapi.json>`.
- CATS dry-run for each enabled block where stable. If a CATS dry-run fails due
  to a tool-level error, capture the exit code and stderr, then fall back to
  command-construction tests plus a tiny live run against the isolated server.
- One live `public-read` run against an isolated local server.
- One live `auth-read` run after env scrubbing is implemented.
- `python -m pytest` for runner/unit helpers when they exist.
- Bandit over touched executable harness code.
- `git diff --check`.

## Open Questions

- Which CI workflow should eventually own nightly fuzzing artifacts.
- Which CRUD route family should be the first seeded stateful block.
- Whether to fix the vector store OpenAPI examples in the same PR as the first
  harness slice or as a separate cleanup PR.

## Implementation Decision

The first implementation slice should start uvicorn itself by default. This
keeps credential isolation, temporary database paths, port allocation, and
cleanup under the runner's control. A later `--server-url` mode can support
manual runs against an already-started server, but it should be opt-in because
the runner cannot fully verify an external server's environment.
