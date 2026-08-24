# Deploying Standalone HTML Presentations

Standalone HTML presentation generation is disabled by default. Saved standalone documents are executable, untrusted, opaque text. The server validates their storage shape but does not sanitize them into safety, and no tldw surface previews or executes them.

## Before Enabling

1. Back up every per-user `Slides.db` and the Jobs database.
2. Upgrade the server and allow the forward migration to Slides schema version 2 to complete.
3. Verify `html5lib` and `tinycss2` are present through the normal project installation.
4. Configure an environment-only digest keyring.
5. Select one built-in adapter and an exact provider/model/adapter tuple.
6. Keep both generation and egress disabled until capabilities report the expected source-free configuration.

Schema v2 is forward-only for old binaries. Do not start an old server binary against a migrated Slides database and do not attempt a database downgrade. Restore the pre-upgrade backup if rollback requires the old binary.

## Generation Configuration

Configure `[SlidesStandaloneHtml]` in `tldw_Server_API/Config_Files/config.txt`. The default is:

```ini
[SlidesStandaloneHtml]
enabled = false
egress_enabled = false
default_provider =
default_model =
default_adapter_id =
allowed_targets_json = []
```

Enablement requires the default `(provider, model, adapter_id)` tuple to appear exactly in `allowed_targets_json`. Wildcards, request-time overrides, custom adapters, endpoint overrides, proxies, routers, fallbacks, and disabled TLS verification are rejected. Endpoint identities come only from the closed built-in adapter catalog.

| Adapter ID | Provider | Fixed endpoint identity |
| --- | --- | --- |
| `openai_official_chat_v1` | `openai` | `https://api.openai.com:443/v1/chat/completions` |
| `anthropic_official_messages_v1` | `anthropic` | `https://api.anthropic.com:443/v1/messages` |
| `llamacpp_loopback_chat_v1_ipv4` | `llama.cpp` | `http://127.0.0.1:8080/v1/chat/completions` |
| `llamacpp_loopback_chat_v1_ipv6` | `llama.cpp` | `http://[::1]:8080/v1/chat/completions` |
| `ollama_loopback_chat_v1_ipv4` | `ollama` | `http://127.0.0.1:11434/v1/chat/completions` |
| `ollama_loopback_chat_v1_ipv6` | `ollama` | `http://[::1]:11434/v1/chat/completions` |

For example, an allowlist entry has exactly these keys:

```json
[{"provider":"openai","model":"your-reviewed-model","adapter_id":"openai_official_chat_v1"}]
```

Set `default_provider`, `default_model`, and `default_adapter_id` to that same tuple. Configure the provider credential through the existing provider-secret mechanism. Never place a provider key, endpoint credential, or digest secret in `allowed_targets_json`.

The configurable source, provider-response, timeout, and output-token values can only lower the fixed V1 exposure ceilings. They cannot raise them.

## Digest Keyring

Secrets are environment-only:

- `SLIDES_STANDALONE_HMAC_KEYS_JSON` is a strict JSON object mapping one to four key IDs to canonical unpadded base64url encodings of 32 random bytes.
- `SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID` names exactly one configured key.

The database registry stores source-free key IDs and lifecycle timestamps, never secret bytes. New receipts use only the current key. A retiring key must remain available for at least 32 days and cannot be removed until a complete fenced dormant-database sweep proves no unexpired receipt references it. Startup or reload fails generation closed if any active/retiring registry key is missing locally.

Generate and distribute key material with your normal secret manager. Do not put it in `config.txt`, a database, logs, shell history, or documentation.

## Effective Limits

The V1 hard ceilings include:

- 4 MiB standalone generation request body
- 200,000 resolved source characters and 50,000 source tokens
- 500 audience characters
- 100 note IDs and RAG `top_k` up to 100
- 8 MiB provider success envelope
- provider connect, read, and overall timeouts default to 10, 120, and 180
  seconds respectively and are configurable downward only
- provider output tokens default to 16,384 with a fixed ceiling of 32,768
- 1 MiB standalone document, source save, and draft attachment
- 1 through 30 slides
- HTML nesting depth 128
- newest 25 standalone entity snapshots per presentation by default, configurable downward only

The capability response reports effective limits after downward configuration clamping.

## Worker And Readiness

The supported lifecycle is the in-process startup service. It registers the `presentation.generate` Jobs handler, validation pool, and reconciler. Do not invent a separate standalone worker CLI or health endpoint.

Use the authenticated, source-free endpoint below for readiness and safe reason codes:

```text
GET /api/v1/slides/capabilities
```

Generation is ready only when `generation_modes.standalone_html.enabled` is true. Relevant disabled reasons include feature or egress disablement, invalid/missing default target, missing prompt asset, missing digest key, unavailable worker, overloaded reconciler, or unavailable validator. Provider health is deliberately not probed by capabilities; provider outages become bounded job failures.

The reconciler checks active Jobs owners at startup and at least once per
minute, and completes the dormant-database sweep at least every 15 minutes.
Generation inputs become inaccessible and expire after 24 hours; the bounded
running sweep physically purges them within the following 15 minutes. Terminal
receipt metadata is retained for 30 days. Saved presentations remain until the
user deletes them.

Provider calls are at-least-once. A worker crash after the provider responds but before the Slides commit can repeat a provider call and its cost. Idempotent transport replay and post-commit reconciliation do not create a second committed presentation.

## CORS And Reverse Proxies

For a separate-origin WebUI, preserve the configured exact origin allowlist and credentials policy. Do not use wildcard credentialed CORS. Preflight must allow the normal authenticated headers plus:

- `Idempotency-Key`
- `If-Match`
- `X-Slides-Accept-Content-Kinds`

Applicable responses must expose `Content-Disposition`, `ETag`, `Last-Modified`, `Retry-After`, `Content-Length`, and the existing request/trace headers. Preserve `Vary: Origin` and content-kind/auth variation.

Do not enable request/response body logging, tracing capture, or error-report payload capture on source-bearing Slides routes. Logs and metrics may include only bounded source-free status, kind, provider/model/adapter identity, duration, byte/slide counts, and stable error codes.

## MCP WebSocket Deployment

Slides remains available through supported non-WebSocket MCP transports, including HTTP, subject to its normal policy. Unguarded WebSocket transports omit/reject Slides tools because they cannot prove the required pre-materialization source guard.

The only supported guarded MCP server launcher that advertises Slides over
WebSocket is:

```bash
python -m tldw_Server_API.scripts.run_server_guarded_mcp
```

It installs `GuardedSlidesWebSocketProtocol` and disables WebSocket compression. A query parameter or request header cannot forge the guarded marker. Do not advertise Slides over a different WebSocket launcher.

## Rollout

1. Deploy the new binary with `enabled = false` and `egress_enabled = false`.
2. Complete backups and schema-v2 migration checks.
3. Configure the keyring and exact allowed target.
4. Verify source-free capabilities and reason codes.
5. Enable the feature flag while leaving egress disabled; confirm saved standalone reads and structured Slides compatibility.
6. Enable egress deliberately and submit a bounded test generation.
7. Verify Jobs completion, source-free logs, strong ETags, attachment headers, and per-user isolation.
8. Expand access gradually while monitoring the source-free capability
   response, ordinary Jobs status, and existing source-free application logs
   and diagnostics. There is no standalone validator or reconciler metric or
   health endpoint.

The browser extension remains metadata-only for standalone projects and hands editing to the canonical WebUI. It must never become a source cache or execution surface.

## Emergency Disable And Rollback

1. Set `egress_enabled = false` first to stop new provider calls.
2. Set `enabled = false` to close new standalone generation admission.
3. Drain active work through the normal Jobs and service lifecycle. Do not kill the database during a commit or migration.
4. Confirm `GET /api/v1/slides/capabilities` reports generation disabled.
5. Keep the current binary and schema v2 online for owner-scoped read, edit, version, and attachment access to already-saved documents.

Disabling generation does not delete saved HTML. Never roll back by downgrading the live database or starting an old binary against schema v2. If a full binary rollback is required, stop services and restore the backup made before migration.

## Release Governance

Any PR materially authored or reworked by AI, including code, tests, docs,
configuration, or review fixes, is not merge-ready until the human requester
provides a `Change summary` in their own words explaining what changed and why
those implementation choices were made. See
[`ADR 004: AI-generated PR Change summary gate`](../ADR/004-ai-generated-pr-change-summary-gate.md).
