# Chatbook Tools: Getting Started

This guide shows how to use the new Chatbook Tools features: templated chat dictionaries, slash commands, and the dictionary validator.

## Overview

- Templating: Render dynamic values (date/time, matched text) in chat dictionary replacements using a sandboxed Jinja2 renderer.
- Slash commands: Pre-LLM helpers like `/time`, `/weather`, `/skills`, and `/skill` to enrich context.
- Validator: Lint chat dictionaries for schema, regex, and template issues (CLI + API).

## 1) Templating in Chat Dictionaries

Templating is expression-only Jinja2 with a strict sandbox (no loops/macros/imports).

Enable templating (precedence):
- Global: `CHAT_DICT_TEMPLATES_ENABLED=1` (env) or `enable_templates = true` (config) turns dictionary templating on; `0/false` disables templating entirely, regardless of per-dictionary settings.
- Per-dictionary: when global templating is enabled, a dictionary-level `enable_templates=false` forces pass-through behavior for that dictionary; `enable_templates=true` forces rendering for that dictionary (subject to sandbox/validation).
- Or config.txt:
  ```ini
  [Chat-Templating]
  enable_templates = true
  ```

Supported helpers (subset):
- Date/time: `now(fmt='%Y-%m-%d')`, `today(fmt)`, `iso_now()`, `now_tz(fmt, tz='UTC')`
- Strings: `upper(s)`, `lower(s)`, `title(s)`; filter: `|slugify`
- Context: `matched_text` (current match), `match` (regex SafeMatch: `group()`, `groups()`, `groupdict()`, `start()`, `end()`)

Examples:
- Literal entry replacement:
  - Pattern: `today`
  - Replacement: `It is {{ now('%B %d') }}.` → “It is November 11.”
- Regex per‐match replacement:
  - Pattern: `/User:(\w+)/`
  - Replacement: `Hello, {{ match.group(1) }}!`
- Using matched text:
  - Pattern: `/\bAI\b/`
  - Replacement: `{{ matched_text|lower }}` → “ai”
- Slugifying matched text:
  - Pattern: `/Project:([A-Za-z0-9_\-\s]+)/`
  - Replacement: `{{ match.group(1)|slugify }}` → “my-project-name”

Limits and safety:
- Expression-only: control structures are rejected.
- Output cap: `MAX_TEMPLATE_OUTPUT_CHARS` (default: 2000).
- Render timeout: `TEMPLATE_RENDER_TIMEOUT_MS` (default: 250 ms). Note: soft timeout — rendering is not interrupted; only metrics are recorded.
- Random helpers are disabled by default; enable with `CHAT_DICT_TEMPLATES_ALLOW_RANDOM=1`.
- Templates are compiled and cached per replacement string using a small LRU cache keyed by the literal template text, to avoid reparsing on every match.

Optional defaults in config.txt:
```ini
[Chat-Templating]
allow_random = false
allow_external_calls = false
max_output_chars = 2000
render_timeout_ms = 250
default_timezone = UTC

Tip: For deterministic tests when random helpers are enabled, set `TEMPLATES_RANDOM_SEED`. A request-scoped seed (when provided by tests) overrides the global seed; the global seed is mainly for test harnesses or controlled environments.
```

## 2) Slash Commands

Slash commands run before messages reach the LLM. Results are injected as a system message (default), prefixed to the user’s text, or can fully replace the user’s text.

Enabled via:
```ini
[Chat-Commands]
commands_enabled = true
injection_mode = system   # or: preface | replace
commands_rate_limit_user = 10    # per-user per-command RPM
commands_rate_limit_global = 100 # global per-command RPM
commands_max_chars = 300         # max injected chars per command result
require_permissions = false
default_location =        # fallback for /weather
```

Built-in commands:
- `/time [TZ]` → “Current time (America/New_York): 2025-11-10 20:15:00”
- `/weather [location]` → “Boston: 42°F, clear skies” (requires provider config; otherwise “weather unavailable”). Configure `WEATHER_PROVIDER=openweather` and `OPENWEATHER_API_KEY` to enable live weather lookups.
- `/skills [filter]` → lists invocable skills for the current user (optional substring filter on name/description/argument hint).
- `/skill <name> [args]` → executes an invocable skill and injects its output using the configured slash-command injection mode.

Examples:
- `/skills summarize`
- `/skill summarize release notes`

Preface-mode example:
- Input (user): `/time America/Los_Angeles`
- Mode: `injection_mode = preface`
- Final user message text: `[/time] Current time (America/Los_Angeles): 2025-11-10 20:15:00`
  - If arguments are present, they are appended after a blank line: `[/time] ...\n\nAmerica/Los_Angeles`.

Replace-mode example:
- Input (user): `/weather Boston`
- Mode: `injection_mode = replace`
- Final user message text sent to the model: `[/weather] Boston: 42°F, clear skies`

Discovery endpoint:
- `GET /api/v1/chat/commands` → list of commands with `name`, `description`, `required_permission`, `usage`, `args`, `requires_api_key`, `rate_limit`, and `rbac_required` (filtered to commands the current user can invoke).
  - When commands are disabled (`commands_enabled=false`), this endpoint returns an empty list. Clients should fetch this endpoint on each session or page load rather than caching the list long-term, since RBAC and configuration may change which commands are available (e.g., enabling/disabling `/weather`).

Moderation ordering:
- Injected system parts bypass user-input moderation but are logged and audited with provenance metadata.
- Output moderation policy can still apply downstream.

Note: Slash commands are pre-LLM conveniences and are separate from LLM tool_calls/MCP tools. Existing tool calling flows are unaffected.

## 3) Dictionary Validator (CLI + API)

Validate structure, regex safety, and template syntax before importing.

CLI:
```bash
python -m tldw_Server_API.app.core.Chat.validate_dictionary --file path/to/dict.json --strict
```

API:
- `POST /api/v1/chat/dictionaries/validate`
  ```json
  {
    "data": {
      "name": "Example",
      "entries": [
        {"type": "literal", "pattern": "today", "replacement": "It is {{ now('%B %d') }}."},
        {"type": "regex", "pattern": "User:(\\w+)", "replacement": "Hello, {{ match.group(1) }}!"}
      ]
    },
    "schema_version": 1,
    "strict": false
  }
  ```

Response:
```json
{
  "ok": true,
  "schema_version": 1,
  "errors": [],
  "warnings": [{"code": "regex_ambiguous", "field": "entries[1].pattern", "message": "…"}],
  "entry_stats": {"total": 2, "regex": 1, "literal": 1},
  "suggested_fixes": []
}
```

Notes:
- `probability` in the validator payload is a float in the range `[0.0, 1.0]`.
- Unknown template functions are flagged as warnings by the validator; external calls (e.g., `weather()`) are disabled by default and will be reported accordingly.
 - `strict=false` (API default) means the validator never rejects based solely on warnings; clients receive `errors` and `warnings` and decide how to handle them. `strict=true` is intended for server-side workflows (e.g., imports) where certain error codes (schema/regex/template/output/size) should be treated as fatal.
 - Unknown or unsupported `schema_version` values result in a normal 200 response with a `schema_invalid` error in the payload; 400 is reserved for requests that do not match the `ValidateDictionaryRequest` schema at all.

## 4) Chatbooks Import: Validator Warnings

When importing Chatbooks synchronously via `/api/v1/chatbooks/import`, validator findings for embedded dictionaries are surfaced in the `warnings` array of the response. In strict mode (`CHATBOOKS_IMPORT_DICT_STRICT=1`), dictionaries with fatal validation errors are skipped entirely; imports still complete best-effort, with warnings describing which dictionaries were rejected and why.

## 5) Troubleshooting

- Templating not applied: ensure `CHAT_DICT_TEMPLATES_ENABLED=1` (or config `[Chat-Templating].enable_templates=true`).
- `/weather` says unavailable: set provider keys and `DEFAULT_LOCATION`, or pass a city in the command.
- Validate timeouts: lower dictionary size or disable risky regexes; use the validator to pinpoint issues. The validator enforces an overall time budget and may emit `regex_timeout` for particularly expensive patterns, skipping further checks for those entries while still returning a best-effort report.
