# Chatbook Tools PRD

- Document Owner: tldw_server core team (product + backend)
- Last Updated: 2025-11-11
- Status: Draft (ready for engineering breakdown)

## 1. Overview

This PRD specifies three pragmatic enhancements to Chatbooks and the Chat module:

- Template variables: Render dynamic values (date, user, context) in dictionary replacements and in Chatbook text.
- User-invoked functions: Allow function-style calls from templates and via user-entered slash commands (e.g., `/weather`).
- Validation: Provide a validator (CLI + API) to lint chat dictionaries, including templates and regexes.

The work fits existing architecture and patterns, prioritizing safety and backwards compatibility. It integrates with chat dictionaries, chat orchestration, and Chatbooks import/export without altering LLM provider tool-calling semantics.

## 2. Goals and Non-Goals

### Goals
- Add a sandboxed template renderer with a small, safe function registry.
- Render templates in chat dictionary replacements; optionally expose rendering for Chatbooks content.
- Add a lightweight slash command router that runs before LLM dispatch.
- Provide a dictionary validation tool both as CLI and API; reuse during Chatbooks import.

### Non-Goals
- Replace or modify provider-specific LLM tool-calling behavior.
- Add new database tables or break existing schemas.
- Introduce broad, arbitrary code execution in templates (must remain sandboxed).

## 3. Success Criteria
- Template rendering enabled behind flags; default behavior unchanged for existing content.
- Slash commands functional for `/time` and `/weather` with rate limits and graceful fallbacks.
- Validator detects malformed regex, unsafe/unknown template functions, duplicates, and bad probabilities; integrates with Chatbooks import warnings.
- Unit + integration tests pass; no regressions in existing chat flows.

## 4. Users and Use Cases

### Personas
- Power user: wants dynamic snippets in prompts and dictionaries (e.g., current date).
- Researcher: wants quick contextual commands (weather/time) in chat.
- Admin/operator: wants safe validation before importing shared dictionaries/chatbooks.

### Representative Stories
- “As a user, I can use `{{ now('%B %d') }}` in a replacement so responses mention today’s date.”
- “As a user, I can type `/weather Boston` and receive a short weather context added to my message.”
- “As an admin, I can validate a dictionary before import and see errors/warnings.”

## 5. Functional Requirements

### 5.1 Template Variables
- Engine: Jinja2 SandboxedEnvironment with curated globals/filters; StrictUndefined; expression-only mode (no loops/macros/blocks); no raw Python exposure. Autoescape disabled (plaintext).
- Expression-only enforcement: renderer validates the parsed Jinja AST against an allowlist of expression nodes (e.g., Name, Const, Call, Filter, Getitem, Getattr) and rejects any non-expression/control nodes (e.g., Block, For, Macro, Assign, Import).
- Functions (initial set):
  - Date/time: `now(fmt='%Y-%m-%d')`, `today(fmt)`, `iso_now()`, `now_tz(fmt='%Y-%m-%d', tz='UTC')`
  - Text filters: `upper(s)`, `lower(s)`, `title(s)`, `slugify(s)`
  - Random (optional, gated): `randint(a,b)`, `choice(list)`; optional deterministic seeding in tests
  - User context: `user()` → `{ id, display_name }` only (no email/tokens)
  - Optional provider plug-in: `weather(city=None)` (disabled by default; requires provider config)
- Context source: TemplateContext capturing `user`, `chat` (character, conv), `request_meta` (safe, pre-sanitized subset such as coarse location/timezone but never raw IPs or sensitive headers), `env` (timezone/locale), `extra`.
- Timezone handling: Use `zoneinfo` for timezone resolution with defaults from user profile (if available) or server config. Locale-aware formatting is deferred for now (no Babel requirement); Chatbooks may still specify `metadata.template_timezone`. `metadata.template_locale` is accepted but ignored unless Babel is installed and explicitly enabled later via a feature flag or explicit configuration. All built-in date/time helpers initially format in a fixed (English) locale; any future locale-aware formatting will be feature-flagged.
- Auto-detection: If a replacement contains `{{`, it is treated as a template when the feature flag is on. Occurrences of `{%` are never considered valid for this feature; any template containing `{%` is rejected as a forbidden construct (`template_forbidden_construct`) and must use escaping to emit literal `{%`/`%}`.
- Failure behavior: On template errors, log and fall back to the original text, never blocking chat. Enforce `MAX_TEMPLATE_OUTPUT_CHARS` and per-render timeout.

### 5.2 Usage in Chat Dictionaries
- Injection point: render per match in `apply_replacement_once` within `tldw_Server_API/app/core/Chat/chat_dictionary.py`.
  - Regex entries: use `subn(lambda m: render(entry.content, context | { 'match': m }))`.
    - Safety: templates receive a SafeMatch wrapper (not raw `re.Match`) exposing only `group(idx|name)`, `groups()`, `groupdict()`, `start()`, and `end()`.
    - Allow `{{ match.group(1) }}` and named groups `{{ match.group('name') }}` via the SafeMatch API.
  - Literal entries: expose `matched_text` for the current replacement: `render(entry.content, context | { 'matched_text': matched })`.
  - Optimization: If no template syntax is present in `entry.content`, perform a fast direct replacement.
  - Compilation/caching: Templates are compiled and cached per `entry.content` using a small LRU cache to avoid reparsing on every match; cache size is bounded (default: `TEMPLATE_CACHE_MAX_ENTRIES=256`) and keyed only by the literal template string to prevent unbounded growth from untrusted inputs.
- Flags:
  - `CHAT_DICT_TEMPLATES_ENABLED` (see Configuration; default false in initial rollout)
  - `CHAT_DICT_TEMPLATES_ALLOW_RANDOM` (default: false)
  - `TEMPLATES_ALLOW_EXTERNAL_CALLS` (default: false; required for `weather()`)
- Dictionary-level optional `enable_templates` override (default auto-detect using template syntax).
- Precedence for template toggling:
  - Global: `CHAT_DICT_TEMPLATES_ENABLED=false` disables dictionary templating entirely, regardless of per-dictionary settings.
  - Per-dictionary: when global templating is enabled, a dictionary-level `enable_templates=false` forces pass-through behavior for that dictionary even if template syntax is present; `enable_templates=true` forces rendering for that dictionary (subject to sandbox/validation).
  - Chatbooks: `metadata.template_mode` governs when Chatbook content is rendered (pass-through vs render-on-import/export) but does not override the global on/off switch for dictionary templating; it operates on top of the global/dictionary-level behavior.
- Escaping and literal behavior:
  - To render literal `{{`/`}}` or `{%`/`%}` in output (since block/control tags are not allowed), users can rely on Jinja expression escaping, e.g., `{{ '{{' }}` to emit `{{` and `{{ '}}' }}` to emit `}}`.
  - For dictionaries that are heavy in curly braces and should never be treated as templates, set `enable_templates=false` at the dictionary level so that content is always treated as literal text even when `{{`/`{%` appear.

### 5.3 Usage in Chatbooks
- Expose the same renderer for Chatbooks content fields (notes, prompts, generated docs, dictionaries).
- Manifest metadata hints (optional):
  - `metadata.template_mode`: `pass_through | render_on_export | render_on_import` (default: pass_through)
  - `metadata.template_defaults`: dict of default context values
  - `metadata.template_timezone`: e.g., `"UTC"` or `"America/New_York"`
  - `metadata.template_locale`: e.g., `"en_US"` — stored as metadata only and not used for locale-aware formatting in the initial implementation; any future locale-aware formatting will be explicitly feature-flagged and require Babel or an equivalent locale library.
- Import/export never executes networked template functions unless explicitly allowed by config (disabled by default via `TEMPLATES_ALLOW_EXTERNAL_CALLS=false`).
- Recommended patterns for `template_mode`:
  - `pass_through`: Use for dynamic Chatbooks whose content should be evaluated at use time (e.g., daily briefing templates, rotating prompts).
  - `render_on_export`: Use for snapshot-style exports where you want to freeze all template fields into concrete text at export time (e.g., archival copies).
  - `render_on_import`: Use sparingly for migration flows where you want to normalize incoming Chatbooks immediately; avoid for content that should remain dynamic.

### 5.4 Slash Commands
- Command router with simple registry: `register("weather", fn)`, `register("time", fn)`. Commands are registered by bare name; the router matches user input starting with `/` and strips the leading slash before lookup (e.g., `/weather` → `"weather"`).
- Hook in `tldw_Server_API/app/core/Chat/chat_orchestrator.py` prior to dictionary processing. If message matches `^/(\w+)(?:\s+(.*))?$`, resolve and execute.
- Injection mode: Prefer injecting the result as a separate `system` message part to keep user message intact and to simplify moderation/auditing; allow opt-in behavior to preface user text; optionally replace the user's message entirely with the command result. Controlled by `CHAT_COMMAND_INJECTION_MODE=system|preface|replace` (default `system`).
- Location resolution priority: request-provided lat/long > user profile location > `DEFAULT_LOCATION` config. No IP-based geolocation is used unless explicitly enabled by `ALLOW_IP_GEOLOCATION=true` with a configured `GEO_PROVIDER`.
- Weather provider: requires configured API key; otherwise `/weather` returns a short “weather unavailable” notice. Provider calls use HTTPX, strict timeouts, and are fully mockable in tests.
- Rate limiting: Reuse the existing chat rate limiter with per-command sub-buckets; enforce per-user and global caps.
- RBAC: Commands are permission-gated using per-command privileges (e.g., `chat.commands.time`, `chat.commands.weather`) and a list privilege (`chat.commands.list`). Enforcement is handled by AuthNZ on a per-user basis.
- Discovery endpoint: `GET /api/v1/chat/commands` returns available commands and brief help (see 7.1 for response fields). Clients should treat this endpoint as the source of truth for each session, since RBAC and configuration changes may add or remove commands between requests.
- Template external calls vs commands: `TEMPLATES_ALLOW_EXTERNAL_CALLS` only affects template functions like `weather()` used inside templates; it does not enable or disable slash commands such as `/weather`, which are governed by `CHAT_COMMANDS_ENABLED`, AuthNZ/RBAC, and provider configuration.

### 5.5 Validation (Chat Dictionaries)
- CLI and API validator with checks:
  - Schema shape vs `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
  - Regex validation for `type=regex`, with safe-regex heuristics and optional match-time timeouts (when the third-party `regex` module is available) to flag catastrophic backtracking risks
  - Template dry-run parse using the sandbox; unknown functions/filters → warnings/errors; enforce expression-only mode
  - Duplicates, empty patterns, invalid probabilities, max_replacements bounds
- Outputs structured report with `errors` (code, field, message), `warnings`, `suggested_fixes`, and `entry_stats` (regex/literal counts).
- Chatbooks import (`/api/v1/chatbooks/import`) calls validator for embedded dictionaries; findings go to `ImportJob.warnings`; strict mode rejects import.
- Validation behavior and strictness:
  - `strict=false` (default for API): validation never rejects based solely on warnings; clients receive all `errors` and `warnings` in the response and decide whether to proceed.
  - `strict=true`: certain error codes are treated as fatal for server-side workflows (e.g., Chatbooks import) and may cause rejection or a non-successful import status. Fatal codes include (at minimum): `schema_invalid`, `regex_invalid`, `regex_unsafe`, `regex_timeout`, `template_parse_error`, `template_forbidden_construct`, `template_output_too_large`, `dictionary_entry_too_large`, `dictionary_too_large`.
  - Some codes (e.g., `template_unknown_function`, `template_undefined_name`) may be downgraded to warnings in non-strict mode when the issue can be tolerated; in strict mode they are treated as errors but do not necessarily block import unless explicitly listed as fatal for that workflow.

- Validator workflows (summary):

| Workflow                         | `strict` value passed | Fatal behavior                                                                                                 | Typical usage                                                                                             |
|----------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| CLI (`python -m ...validate`)    | `strict` flag from CLI (`--strict` → true; default false) | When `--strict` is set, the presence of any fatal error code (schema/regex safety/timeout/template parse/forbidden/output/size) MUST cause a process exit code of 1; when `--strict` is not set, the CLI exits 0 unless an internal error occurs (internal errors SHOULD exit with a distinct non-zero code such as 2). | Local linting in dev/CI; treat `--strict` as “fail the build on serious issues”, surface all issues in JSON. |
| API `POST /chat/dictionaries/validate` | `strict` field from request (default false) | Never changes HTTP status: always 200 on successful validation. In `strict=true`, fatal codes are classified as errors (not warnings), but the caller decides whether to block on them. | Online tools, WebUI forms, and API clients that want structured reports without automatic rejection.      |
| Chatbooks import                 | Always calls validator with `strict=false`, paired with `CHATBOOKS_IMPORT_DICT_STRICT` env flag | When `CHATBOOKS_IMPORT_DICT_STRICT` is enabled, any fatal error codes cause the offending dictionary to be skipped entirely; warnings are always non-fatal but are recorded in `ImportJob.warnings`. Import workflows never fail the entire Chatbook solely due to dictionary issues; only the problematic dictionaries are skipped in strict mode. | Importing Chatbooks where malformed dictionaries should not block the whole import but should be skipped when strict mode is requested. |

## 6. Architecture & Components

### New Modules
- `tldw_Server_API/app/core/Templating/template_renderer.py`
  - Sandboxed Jinja environment factory
  - Safe function registry and filters
  - `render(text, context, options)` entrypoint with guardrails

- `tldw_Server_API/app/core/Chat/command_router.py`
  - Registry, dispatcher, rate limiting utilities
  - Built-in commands: `time` (local, invoked as `/time`), `weather` (via provider abstraction, invoked as `/weather`)

- `tldw_Server_API/app/core/Integrations/weather_providers.py`
  - Provider interface (OpenWeather, no-key fallback)
  - HTTPX client with timeouts, mockable

- `tldw_Server_API/app/core/Chat/validate_dictionary.py`
  - CLI interface (`python -m ... validate_dictionary --file <path> [--strict]`)
  - Shared validation routines reused by API endpoint and Chatbooks import

### Touchpoints (Minimal Changes)
- `tldw_Server_API/app/core/Chat/chat_dictionary.py`
  - Render per match inside `apply_replacement_once` with access to `match` (regex) or `matched_text` (literal)

- `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
  - Detect and dispatch slash commands before dictionary processing
  - Inject command result as a separate `system` message part (preferred) or prepend context to user text

- `tldw_Server_API/app/core/Chatbooks` (import path)
  - On import, run validator for embedded dictionaries; attach warnings to `ImportJob.warnings`
  - Respect `metadata.template_mode` during preview/import if configured

## 7. API Surface

### 7.1 List Commands
- Method/Path: `GET /api/v1/chat/commands`
- Response:
  {
    "commands": [
      {
        "name": "time",
        "description": "Show the current time (optional TZ).",
        "required_permission": "chat.commands.time",
        "usage": "/time [timezone]",
        "args": ["timezone"],
        "requires_api_key": true,
        "rate_limit": "per-user 10/min",
        "rbac_required": true
      },
      {
        "name": "weather",
        "description": "Show current weather for a location.",
        "required_permission": "chat.commands.weather",
        "usage": "/weather [location]",
        "args": ["location"],
        "requires_api_key": true,
        "rate_limit": "per-user 10/min",
        "rbac_required": true
      }
    ]
  }
  - The list is filtered per user permissions and deployment configuration. Only commands the user can invoke are returned; each entry still includes `required_permission` for client display along with optional metadata such as `usage`, `args`, `requires_api_key`, `rate_limit`, and `rbac_required` for richer client UX. Deployments may choose to hide commands whose backing providers are not configured (e.g., omit `weather` when no weather provider or API key is set) to avoid advertising unusable commands; alternatively, they may expose such commands with a configurable “unavailable” message to guide users to contact an administrator.

### 7.2 Validate Dictionary
- Method/Path: `POST /api/v1/chat/dictionaries/validate`
- Request: `{ "data": { ... }, "schema_version": 1, "strict": false }`
- Response:
  - {
      "ok": true,
      "schema_version": 1,
      "errors": [
        {"code": "regex_invalid", "field": "entries[3].pattern", "message": "Unclosed group"}
      ],
      "warnings": [{"code": "template_unknown_function", "field": "replacement", "message": "Unknown function: weather"}],
      "entry_stats": { "total": 25, "regex": 5, "literal": 20 },
      "suggested_fixes": ["escape '.' in pattern 2"],
      "partial": false,
      "partial_reason": null
    }
  - Status codes:
    - 200 on validation completion (with errors/warnings payload), including when validation finds issues (i.e., `ok=false` or non-empty `errors`/`warnings`).
    - 200 with `schema_invalid` in `errors` when an unknown or unsupported `schema_version` is provided.
    - 400 only for malformed request payload (e.g., request body not matching `ValidateDictionaryRequest` schema).
  - Partial results:
    - The response includes a `partial` flag (bool) and optional `partial_reason` field (e.g., `"max_entries"`, `"timeout"`) to indicate when validation short-circuited due to `CHAT_DICT_VALIDATE_MAX_ENTRIES` or time-budget limits while still returning a best-effort report.

## 8. Configuration

- `CHAT_DICT_TEMPLATES_ENABLED` (bool, default false initially; flipped to true after bake-in)
- `CHAT_DICT_TEMPLATES_ALLOW_RANDOM` (bool, default false)
- `TEMPLATES_ALLOW_EXTERNAL_CALLS` (bool, default false)
- `CHAT_COMMANDS_ENABLED` (bool, default true)
- Rate limits (reuse existing limiter semantics):
  - `CHAT_COMMANDS_RATE_LIMIT_USER` (e.g., `10/min`)
  - `CHAT_COMMANDS_RATE_LIMIT_GLOBAL` (e.g., `100/min`)
- `DEFAULT_LOCATION` (string, optional, e.g., `"San Francisco, CA"`)
- Weather provider: `WEATHER_PROVIDER`, `OPENWEATHER_API_KEY`, timeouts/retry knobs
 - `MAX_TEMPLATE_OUTPUT_CHARS` (int, default 2000; hard cap on rendered output length)
 - `TEMPLATE_RENDER_TIMEOUT_MS` (int, default 250)
 - `TEMPLATES_RANDOM_SEED` (optional, for deterministic tests; a request-scoped seed overrides the global seed when present, and the global seed is primarily intended for test harnesses or controlled environments)
 - `TEMPLATE_DEFAULT_TZ`, `TEMPLATE_DEFAULT_LOCALE` (fallbacks; locale currently unused unless Babel is enabled)
 - `TEMPLATE_CACHE_MAX_ENTRIES` (int, default 256; max number of distinct compiled templates cached per process)
 - `CHAT_DICT_VALIDATE_TIMEOUT_MS` (int, default 500; overall time budget per validation request)
 - `CHAT_DICT_VALIDATE_MAX_ENTRIES` (int, default 1000; maximum number of entries processed per validation request before short-circuiting with best-effort results)
 - `CHAT_COMMANDS_MAX_CHARS` (int, default 300; max size for injected system part; larger results are truncated)
 - `CHAT_COMMAND_INJECTION_MODE` (`system`|`preface`|`replace`, default `system`)
 - `WEATHER_UNITS` (`metric`|`imperial`, default `metric`), `WEATHER_LANG` (default `en`)
 - `ALLOW_IP_GEOLOCATION` (bool, default false), `GEO_PROVIDER` (optional; none used unless configured)
 - `CHATBOOKS_IMPORT_DICT_STRICT` (bool, default false; when true, Chatbooks import skips dictionaries with fatal validation errors instead of importing them with warnings only).

## 9. Security & Privacy

- Sandboxed templates; no attribute traversal beyond provided wrappers; no `import`, no filesystem/network unless explicitly allowed.
- External calls (e.g., weather) disabled by default for templates; commands use short timeouts and strict rate limits.
- Avoid logging sensitive content; log only metrics and minimal context (e.g., command name, city tokenized). The `user()` function only exposes `{id, display_name}`.
- IP-based geolocation (when explicitly enabled via `ALLOW_IP_GEOLOCATION` and `GEO_PROVIDER`) is used only for transient resolution; raw IPs and geo provider responses are not persisted beyond request processing. Only coarse, non-PII location strings may be stored in logs or metadata.
- Respect existing AuthNZ modes (single-user API key, JWT multi-user); use RBAC for tool/command endpoints if necessary. The new `GET /api/v1/chat/commands` and `POST /api/v1/chat/dictionaries/validate` endpoints require the same AuthNZ as other chat endpoints and are permission-gated via existing RBAC privileges (e.g., `chat.commands.list`, `chat.dictionaries.validate`).

## 10. Observability

- Metrics counters/histograms (registry names):
  - Templates
    - `template_render_success_total{source=dict|chatbook}`
    - `template_render_failure_total{source=dict|chatbook,reason=parse|exception}`
    - `template_render_timeout_total{source=dict|chatbook}`
    - `template_output_truncated_total{source=dict|chatbook}`
    - `template_render_duration_seconds{source=dict|chatbook}` (histogram)
  - Commands
    - `chat_command_invoked_total{command,status=success|error|rate_limited|denied}`
    - `chat_command_errors_total{command,reason=exception|permission_denied|rate_limited}`
    - Legacy log counters remain (`chat_command_invoked`, `chat_command_error`) for log-based dashboards.
  - Validator
    - `chat_dictionary_validate_requests_total{strict}`
    - `chat_dictionary_validate_errors_total{code}`
    - `chat_dictionary_validate_warnings_total{code}`
    - `chat_dictionary_validate_duration_seconds{strict}` (histogram)
- Metrics label cardinality:
  - `command` is drawn from a fixed, small registry of known command names (e.g., `time`, `weather`) to avoid high-cardinality series.
  - `code` is drawn from a fixed enum of validation error/warning codes (see section 22), not free-form strings.
- Loguru structured logs around failures and timeouts.

## 11. Performance & Limits

- Template render cost is O(text length) and negligible; enforce a max render time and output size (configurable hardcaps) to remain safe.
- Commands enforce:
  - Max concurrency per user
  - Per-invocation timeout (defaults: `/time` 100ms, `/weather` 1500ms)
  - Bounded result size (short summaries; truncated to `CHAT_COMMANDS_MAX_CHARS`)
- Validator performance:
  - The validator enforces an overall time budget per request (e.g., a few hundred milliseconds, configurable) and may cap the maximum number of entries processed per call to prevent abuse.
  - When regex safety checks exceed the match-time budget for a given entry, the validator emits a `regex_timeout` issue for that entry and skips further expensive checks for it (and, if necessary, for remaining entries) while still returning a best-effort report. In these cases, the response sets `partial=true` and provides a short `partial_reason` enum to signal that only a subset of entries was fully validated.

## 12. Error Handling

- Template rendering errors: log + fallback to original text. Client applications (including the WebUI) may optionally surface a subtle warning (e.g., “template rendering failed; using raw text”) when they receive validator issues or detect repeated template failures, to help power users debug misconfigurations without disrupting normal chat flow.
- Moderation ordering: Command-injected context is added as a separate `system` message part, bypasses user-input moderation, and is logged/audited with explicit system-origin metadata; the resulting conversation (including injected parts) remains subject to any downstream output moderation applied by the deployment. When `CHAT_COMMAND_INJECTION_MODE=replace`, the synthetic message is still tagged with system-origin metadata for audit and moderation purposes.
- Command errors/timeouts: prepend a short notice (optional) or silently skip; never block the chat flow. `/weather` yields a short “weather unavailable” notice when provider is not configured.
- Audit metadata on injected parts includes: `{ origin: 'system', cmd: '/name', duration_ms, status: 'success'|'fallback'|'error' }`. Injected content is truncated to `CHAT_COMMANDS_MAX_CHARS`.
- Validator: returns structured errors; Chatbooks import surfaces warnings and rejects only in strict mode or on fatals.

## 13. Testing Strategy

- Unit Tests
  - TemplateRenderer: date functions, filters, sandbox restrictions, random gating.
  - Dictionary flow: replacement with `{{ now('%B %d') }}` renders; disabled when flag off.
  - Regex compile failures: validator flags appropriately.
  - Command router: `/weather` happy path with mocked provider; rate limit; timeout path.

- Integration Tests
  - `/api/v1/chat/dictionaries/process` with `enable_templates=true` returns rendered output.
  - `/api/v1/chat/completions` with message `/weather Boston` injects context (mock provider) and completes.
  - Chatbooks import populates `ImportJob.warnings` when dictionary issues are found.
  - Deterministic random: with seeding enabled, `randint/choice` produce stable outputs in tests.

## 14. Rollout Plan

### Phase 1
- Implement TemplateRenderer and integrate into chat dictionary replacements (date/time + basic filters).
- Add CLI validator (no API endpoint yet).

### Phase 2
- Add command router with `/time` and stub `/weather` (mock provider); `GET /api/v1/chat/commands` endpoint.
- Add API endpoint `POST /api/v1/chat/dictionaries/validate`.

### Phase 3
- Wire validator into Chatbooks import; add manifest `metadata.template_mode` and optional render-on-import.
- Optionally register `weather()` template global, gated by `TEMPLATES_ALLOW_EXTERNAL_CALLS`.

## 15. Backward Compatibility & Migration

- Default behavior remains unchanged: templates are a no-op unless flags are enabled or template syntax is present. Initial rollout ships with templating disabled by default; after bake-in, default may be flipped on.
- Slash command router only acts for messages beginning with `/`.
- No changes required to existing dictionaries or chatbooks unless adopting new features.

## 16. UX Notes (WebUI)

- Show available commands via `GET /api/v1/chat/commands` and a small help popover.
- Optional “Render preview” toggle when viewing/editing dictionaries or chatbook previews.
- Surface validator warnings inline during dictionary import/edit.
- On each session or page load, fetch `GET /api/v1/chat/commands` to populate command hints; do not cache the list long-term, since RBAC rules and configuration may change which commands are available.

## 17. Open Questions

- Should template rendering also apply to the final composed user message globally, or only to dictionary replacements? Recommendation: replacements only to avoid surprises.
- Do we allow user-defined custom functions via plugin mechanism? If yes, behind admin-only config.
- Preferred default weather provider and location determination policy; privacy expectations for location data.

For the initial implementation, adopt the recommendations above as defaults: template rendering applies only to dictionary replacements (not global user messages), no user-defined custom functions are supported, and OpenWeather (when configured) is the default weather provider. Alternative policies can be explored as follow-up iterations.

## 18. Risks & Mitigations

- Risk: Template engine misuse to attempt code execution → Mitigation: Jinja sandbox, restricted globals, tests.
- Risk: Command-induced latency → Mitigation: short timeouts, strict rate limits, non-blocking fallback.
- Risk: Unexpected message changes due to templates → Mitigation: off by default; explicit opt-in; clear docs.

## 19. Dependencies

- Jinja2 (runtime) for templating (already common in Python stacks; add to extras if needed).
- HTTPX for provider calls (already present).
- Existing AuthNZ, rate limiting, and metrics modules.

## 20. Example Snippets

### Dictionary Template Example
```
current_date: |
  The current day is {{ now('%B %d') }}.
```

### Regex Capture Example
```
pattern: "price\\s+(\\w+)"
replacement: "The product {{ match.group(1)|upper }} is on sale today ({{ now_tz('%b %d', tz='America/New_York') }})."
```
- Regex patterns are plain Python regular expressions (no leading or trailing `/.../` delimiters); the example above uses a simple capture group for a product name.

### Slash Command Example
- Input: `/weather Boston`
- Result (prepended context):
  "[Context: Weather for Boston, MA — 68°F, clear skies]"
  - In `GET /api/v1/chat/commands`, the corresponding command entry uses `name: "weather"` (without the leading `/`). The router matches user input starting with `/` and strips the leading slash before lookup, so commands are invoked as `/name` but registered and exposed as bare `name` values.

## 21. References

- Chat Orchestrator: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
- Chat Dictionary: `tldw_Server_API/app/core/Chat/chat_dictionary.py`
- Chat Dictionary Schemas: `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
- Chatbooks Models: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Tools/MCP endpoints: `tldw_Server_API/app/api/v1/endpoints/tools.py`
- Existing Chatbooks PRD: `Docs/Product/Chatbooks_PRD.md`

## 22. Validator Error Taxonomy

Each error/warning includes `code`, `field`, and `message`. Common codes:

- schema_invalid: Request or dictionary shape does not match schema_version.
- unknown_field: Field not recognized/allowed by schema.
- empty_pattern: Pattern is empty or whitespace only.
- duplicate_pattern: Duplicate pattern detected (same type/scope).
- probability_out_of_range: Probability not within [0.0, 1.0].
- max_replacements_invalid: Invalid max_replacements (negative or exceeds limit).
- regex_invalid: Regex failed to compile.
- regex_unsafe: Pattern exhibits catastrophic backtracking risk (heuristic check).
- regex_timeout: Regex validation exceeded match-time budget (only when optional `regex` module is available).
- template_parse_error: Template failed to parse (expression-only mode).
- template_forbidden_construct: Non-expression or blocked syntax used (e.g., loops, macros, blocks).
- template_undefined_name: Template referenced an undefined variable, function, or filter (StrictUndefined).
- template_unknown_function: Unknown function or filter referenced.
- template_external_calls_disabled: External function used but external calls are disabled.
- template_output_too_large: Rendered output exceeds MAX_TEMPLATE_OUTPUT_CHARS.
- dictionary_entry_too_large: Single dictionary entry exceeds configured size limit.
- dictionary_too_large: Dictionary exceeds configured entry/size limits.

Warnings typically reuse the same codes with reduced severity (e.g., `template_unknown_function`) when the issue can be tolerated under non-strict mode. The response payload includes `errors`, `warnings`, `entry_stats`, and `suggested_fixes` for actionable remediation.

## 23. Implementation Plan

This implementation plan breaks delivery into clear engineering stages with flags, file-level changes, and tests. It aligns with the Rollout Plan while adding concrete execution details and acceptance criteria.

### Stage 1 — Template Renderer Core (Sandbox)
- Files
  - Add: `tldw_Server_API/app/core/Templating/template_renderer.py`
    - Sandboxed Jinja environment (StrictUndefined, expression-only, no macros/blocks, no autoescape)
    - Functions: `now`, `today`, `iso_now`, `now_tz`; `upper/lower/title/slugify`; gated `randint/choice` (seed support)
    - Context builder: accepts `user`, `chat`, `request_meta`, `env` (tz/locale), `extra`
    - Options: `allow_random`, `allow_external_calls`, `max_output_chars`, `timeout_ms`, `cache_max_entries`
- Config
  - Wire flags into central config: `CHAT_DICT_TEMPLATES_ENABLED`, `CHAT_DICT_TEMPLATES_ALLOW_RANDOM`, `TEMPLATES_ALLOW_EXTERNAL_CALLS`, `MAX_TEMPLATE_OUTPUT_CHARS`, `TEMPLATE_RENDER_TIMEOUT_MS`, `TEMPLATE_CACHE_MAX_ENTRIES`, `TEMPLATE_DEFAULT_TZ`, `TEMPLATE_DEFAULT_LOCALE`, `TEMPLATES_RANDOM_SEED`
- Tests
  - Unit: `tldw_Server_API/tests/Chat_NEW/unit/test_template_renderer.py`
    - Rendering basics, timezone, locale, strict undefined, output cap, timeout, random gating and deterministic seeding, AST allowlist enforcement (expression-only), `{%`/block-tag rejection, and literal escaping behavior
- Acceptance
  - All helper functions behave; sandbox rejects forbidden constructs (including any `{%` control tags); no side effects or global state

### Stage 2 — Dictionary Integration (Per-match Rendering)
- Files
  - Update: `tldw_Server_API/app/core/Chat/chat_dictionary.py`
    - In `apply_replacement_once`, render replacements per match
      - Regex: `subn(lambda m: render(entry.content, ctx|{'match': m}))`
      - Literal: expose `matched_text` and render per replacement
    - Add fast path when no template syntax present (based on `{{` detection only)
    - Build per-call template context (user/chat/env if available via caller)
- Config
  - Respect `CHAT_DICT_TEMPLATES_ENABLED`, dictionary-level `enable_templates`, and random/external flags
- Tests
  - Unit: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_dictionary_templates.py`
    - Regex groups and named groups work; literal matched_text works; global and per-dictionary template flags behave as specified; disabled flag is no-op
- Acceptance
  - No change in behavior when feature flag off; measurable performance impact near-zero on non-templated entries

### Stage 3 — Validator (Library + CLI)
- Files
  - Add: `tldw_Server_API/app/core/Chat/validate_dictionary.py`
    - Public function `validate_dictionary(data, schema_version, strict=False)` returning structured report
    - CLI entrypoint: `python -m tldw_Server_API.app.core.Chat.validate_dictionary --file <path> [--strict]` (optional console script alias `tldw-dict-validate`)
    - Safe-regex heuristics + optional match-time timeout using the third-party `regex` module (if installed); template parse in expression-only mode; `partial`/`partial_reason` reporting when validation short-circuits due to time/entry limits
- Tests
  - Unit: `tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py`
    - Schema violations, regex invalid/unsafe/timeout, template parse errors, expression-only enforcement (`template_forbidden_construct`), warnings in non-strict mode vs errors in strict mode, `partial`/`partial_reason` behavior when limits are hit, and CLI exit codes for `--strict` vs non-strict and internal-error paths
- Acceptance
  - CLI produces JSON summary with error taxonomy; when invoked with `--strict`, any fatal error codes result in exit code 1, non-fatal issues exit 0, and internal errors use a distinct non-zero code (for example, 2); API responses populate `partial`/`partial_reason` correctly when validation short-circuits

### Stage 4 — Command Router Core
- Files
  - Add: `tldw_Server_API/app/core/Chat/command_router.py`
    - Registry: register(name, handler, rbac, rate_limit)
    - Built-ins: `/time` (local, tz-aware), `/weather` (calls integration; disabled without key)
    - Rate limit hooks: per-user and global caps; short execution timeouts
  - Add: `tldw_Server_API/app/core/Integrations/weather_providers.py` (provider interface + stub)
  - Update: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
    - Parse `^/(\w+)(?:\s+(.*))?$` before dictionary processing
    - Execute command; inject result as a separate `system` message part (preferred) or preface user text (config)
    - Tag injected part with metadata for auditing
- Config
  - `CHAT_COMMANDS_ENABLED`, `CHAT_COMMANDS_RATE_LIMIT_USER`, `CHAT_COMMANDS_RATE_LIMIT_GLOBAL`, `DEFAULT_LOCATION`, weather provider keys/timeouts
- Tests
  - Unit: `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`
    - Parsing, RBAC gating, rate limiting, timeouts; `/weather` w/ mock provider; system-part injection
  - Integration: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_injection.py`
- Acceptance
  - Commands execute or degrade gracefully; no blocking on provider failures; moderation ordering preserved

### Stage 5 — API Endpoints + Chatbooks Integration
- Files
  - Update/Add: Chat endpoints (or new router)
    - `GET /api/v1/chat/commands`: list commands with `usage`, `args`, `requires_api_key`, `rate_limit`, `rbac_required` (RBAC-filtered)
    - `POST /api/v1/chat/dictionaries/validate`: accepts `data`, `schema_version`, `strict`; returns structured report with `partial`/`partial_reason`
  - Update: Chatbooks import flow to invoke validator on embedded dictionaries; always call validator with `strict=false`; append findings into `ImportJob.warnings`; consult `CHATBOOKS_IMPORT_DICT_STRICT` to decide whether dictionaries with fatal errors are skipped while allowing the rest of the Chatbook to import
- Tests
  - Integration: endpoints contract tests; Chatbooks import path populates warnings and respects strict failures
- Acceptance
  - Endpoints pass schema validation and RBAC; validator endpoint returns `partial`/`partial_reason` as specified; Chatbooks import produces validator results without performance regressions and never fails the entire import solely due to dictionary validation errors (problematic dictionaries are skipped when `CHATBOOKS_IMPORT_DICT_STRICT=true`)

### Stage 6 — Hardening, Metrics, Docs
- Files
  - Add metrics counters/histograms and log lines as specified
  - Docs updates: `Docs/API-related/Chatbook_Features_API_Documentation.md` and `Docs/Published/API-related/Chatbook_Features_API_Documentation.md` (append endpoints and tools interplay), `Docs/Product/Chatbooks_PRD.md` alignment notes, WebUI notes, `.env.example`/config.txt knobs
- Tests
  - Smoke: run selected integration flows with flags on/off
- Acceptance
  - Feature flags correctly gate behavior; metrics present; documentation synced with implementation

### Rollout & Backout
- Default flags: templating disabled initially (flip on after bake-in), commands enabled (weather off without key), validator available via CLI; API validator added in Stage 5
- Backout: disable via flags; code paths preserve legacy behavior when off

### Ownership & Review
- Security review: sandbox restrictions, RBAC, rate limits
- Performance check: benchmark dictionary processing with/without templating on typical inputs
- QA sign-off: end-to-end flows for commands, templating, and validator/APIs
