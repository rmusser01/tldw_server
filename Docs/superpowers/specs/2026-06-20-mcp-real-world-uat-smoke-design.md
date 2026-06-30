# MCP Real-World UAT Smoke Scenario Design

## Goal

Add a second MCP smoke scenario that exercises realistic tool/action use before
PR #2415 is marked ready. The existing `baseline` scenario remains fast and
protocol-focused. The new `real-world` scenario validates isolated artifact
handling, multi-tool chaining, mounted-server configurability, and optional
real LLM provider calls behind explicit environment gates.

## Design Review Findings Folded In

1. Keep `baseline` unchanged. A heavier workflow would make simple transport
   checks slower and harder to run in CI.
2. Make artifact use explicit and isolated. The scenario must create or use one
   artifact root per run, write only under that root, and report only redacted
   or root-relative paths.
3. Do not assume mounted tldw_server exposes filesystem tools. The standalone
   fixture can provide deterministic artifact tools, but mounted/live runs need
   configurable tool names and arguments.
4. Real LLM calls must be impossible by accident. They require an explicit
   provider option plus an environment variable name. Missing credentials are a
   skip in best-effort mode and a controlled failure in strict mode.
5. Verify behavior, not prose. The LLM step should validate a small structural
   signal, bounded response size, and redaction behavior rather than exact model
   wording.
6. Keep reports safe. No absolute paths, raw artifact bodies, or API keys should
   appear in JSON reports or console summaries.
7. Wire artifact roots per transport. In-process runs can pass artifact state
   directly. Live fixture and stdio fixture servers need an env-provided
   artifact root because the client process cannot mutate an already-running
   server runtime.
8. Treat mounted artifact UAT as same-host by default. If the target server is
   remote or has different path policy, operators must provide tool templates
   that seed/read/write artifacts through MCP rather than relying on client-side
   filesystem seeding.

## Scenario Shape

`mcp-unified-smoke <transport> --scenario real-world ...` runs these ordered
steps:

1. `initialize`.
2. `tools/list`, capturing available tool names.
3. Seed an artifact input file under an isolated artifact root when the selected
   transport supports same-host artifact setup.
4. Read the artifact through a configured read tool, or through the fixture
   artifact runtime for in-process runs.
5. Retrieve a prompt where available, or skip/fail according to mode.
6. Produce a derived artifact by calling a configured write/action tool.
7. Verify the derived artifact with a read/stat-style tool call.
8. Optionally call a real LLM provider when explicitly enabled.

The report records step outcomes, bounded summaries, artifact-relative paths,
and provider metadata without secrets.

## Artifact Model

Default behavior creates a unique temporary root with:

- `input/product-brief.md`
- `output/` directory for generated artifacts

The scenario accepts `--artifact-dir PATH` for repeatable manual UAT. If the
path exists, the run creates a unique child directory by default. The initial
slice should avoid destructive cleanup so failures preserve evidence. A future
option can add cleanup-on-success.

For live fixture app and stdio fixture runs, the fixture server reads
`MCP_SMOKE_ARTIFACT_ROOT` so the server-side runtime and client-side scenario
agree on the same isolated root. The CLI should set/inherit that env var for
stdio fixture examples and document it for manually started HTTP/WebSocket
fixture apps.

Artifact summaries expose:

- logical name such as `input/product-brief.md`
- byte counts
- hash prefixes when useful
- whether verification succeeded

They must not expose absolute host paths or full file contents.

## Standalone Fixture Runtime

The fixture runtime gains deterministic real-world tools:

- `artifact.read` reads a fixture-managed artifact path.
- `artifact.write` writes a derived artifact under the fixture artifact root.
- `artifact.stat` verifies existence, byte count, and hash metadata.
- `artifact.summarize` derives structured content from the input artifact.

These tools are not intended as user-facing production tools. They are smoke
fixture tools that let in-process, HTTP fixture, WebSocket fixture, and stdio
fixture transports run the same scenario without external services.

For in-process fixture runs, the CLI constructs `SmokeFixtureGatewayRuntime`
with the artifact root directly. For HTTP/WebSocket fixture apps and stdio
fixture servers, the fixture runtime discovers the root from
`MCP_SMOKE_ARTIFACT_ROOT`.

## Mounted/Live Server Configuration

Mounted/live runs can configure tool names and arguments:

- read tool name and argument template
- write/action tool name and argument template
- verification tool name and argument template

The first implementation should provide sensible defaults for fixture tools and
allow mounted tldw_server operators to pass real tool settings when filesystem
or other workspace tools are enabled. Missing required configured tools fail in
strict mode and skip in best-effort mode.

When targeting mounted tldw_server on the same host, `--artifact-dir` may point
inside an allowed workspace path so server-side filesystem tools can access it.
When targeting a remote server, callers should use configured seed/write tools
or pre-existing server-side artifacts; the scenario must not pretend a local
client temp directory is server-visible.

## Real LLM Gate

The optional LLM step is disabled unless all are true:

- `--real-llm-provider` is supplied.
- `--real-llm-api-key-env` is supplied.
- The named environment variable exists and is non-empty.

Initial provider support should target one OpenAI-compatible HTTP path with a
small prompt and bounded response. The scenario must not log request headers,
API keys, full prompts, or full model output. Missing env is:

- skipped in best-effort mode
- a controlled failure in strict mode when the LLM step was explicitly
  requested

## Error Handling

- Artifact root creation failures produce `artifact_setup_failed`.
- Missing server-visible artifact setup produces `artifact_seed_unavailable`.
- Required tool absence produces `required_tool_unavailable`.
- Artifact writes outside the root produce `artifact_path_denied`.
- Verification mismatches produce `artifact_verification_failed`.
- LLM env absence produces `real_llm_env_missing`.
- LLM HTTP/provider failures produce `real_llm_call_failed` with redacted
  detail.

All errors should be represented as normal smoke step failures, not unhandled
tracebacks.

## Testing Strategy

Use TDD:

1. Add failing tests for CLI scenario selection and artifact-root report
   redaction.
2. Add failing tests for the fixture runtime chain producing and verifying an
   output artifact.
3. Add failing tests for strict/best-effort handling of missing configured
   tools.
4. Add failing tests for real LLM env gating that do not perform network calls.
5. Add a mocked OpenAI-compatible HTTP test for the enabled LLM step.

Verification should include focused pytest, compileall, `git diff --check`,
and Bandit on touched production files.

## Non-Goals

- No broad mounted-server filesystem policy changes.
- No automatic discovery of arbitrary mounted filesystem tools in the first
  slice.
- No required external LLM calls in CI.
- No exact model-output assertions.
