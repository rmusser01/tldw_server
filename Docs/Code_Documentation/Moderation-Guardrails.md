Moderation & Guardrails

Overview
- The chat subsystem supports configurable guardrails for inputs and outputs with global settings, per-user overrides, optional categories, and an admin UI.
- Supports non-streaming and streaming modes (SSE). Streaming yields a final `data: [DONE]` on normal completion and emits an SSE error + `[DONE]` when a block occurs mid-stream.

Key Capabilities
- Global policy with input/output enablement and default actions.
- Blocklist with literals or regex and per-pattern actions/replacements.
- Per-user overrides for toggle/actions/redaction and categories.
- Optional categories and built-in PII redaction rules.
- Runtime overrides (admin-controlled) with optional persistence to file.
- Admin UI for blocklist management, overrides listing/editing, runtime settings, and a policy tester.

Configuration ([Moderation] in `tldw_Server_API/Config_Files/config.txt`)
- `enabled` (bool): master switch.
- `input_enabled`, `output_enabled` (bool): phase toggles.
- `input_action`, `output_action`: `block | redact | warn`.
- `redact_replacement` (str): default replacement when redacting.
- `blocklist_file` (path): default `tldw_Server_API/Config_Files/moderation_blocklist.txt`.
- `user_overrides_file` (path): JSON mapping of user_id -> overrides.
- Paths are resolved relative to the project root when not absolute.
- `per_user_overrides` (bool): enable per-user overrides.
- `pii_enabled` (bool): include built-in PII redaction rules (defaults off).
- `categories_enabled` (csv): categories to permit globally (empty = allow all).
- `runtime_overrides_file` (path): default `tldw_Server_API/Config_Files/moderation_runtime_overrides.json`.
- Performance/Safety (optional):
  - `max_scan_chars` (int): scan chunk size per text (default 200000; full text is scanned in chunks).
  - `max_replacements_per_pattern` (int): replacement limit per pattern (default 1000).
  - `match_window_chars` (int): lookahead window to catch matches spanning chunk boundaries (default 4096).
  - `blocklist_write_debounce_ms` (int): debounce window for blocklist writes in milliseconds (default 0=disabled). Useful to coalesce rapid edits from the Web UI.
- ENV overrides: `MODERATION_*` keys mirror the above.

Blocklist Grammar
- Literal: `confidential project`
- Regex: `/secret\s+token/` (case-insensitive by default)
- Regex with flags: `/secret\s+token/imsx` (supported flags:
  - `i` case-insensitive (default already applied)
  - `m` multiline
  - `s` dot matches newline
  - `x` verbose)
- With action:
  - `forbidden term -> block`
  - `/leak(\d+)/ -> redact:[MASK]`
  - `/minor issue/ -> warn`
- With categories (comma-separated suffix; requires whitespace before `#`):
  - `/ssn\b\d{3}-\d{2}-\d{4}/ -> redact:[SSN] #pii`
  - `internal code name #confidential`
  - To include a literal `#` in a pattern or literal term, escape it as `\#`.

Per-user Overrides (user_overrides_file)
- Keys mirror [Moderation] defaults: `enabled`, `input_enabled`, `output_enabled`, `input_action`, `output_action`, `redact_replacement`.
- `categories_enabled`: comma-separated string or list of category names. If set, only rules with intersecting category are active.
- To explicitly clear category gating for a user (allow all categories), set `categories_enabled` to an empty string or empty list.

Categories Behavior
- When `categories_enabled` is provided (globally or per-user), only rules whose `categories` intersect with the enabled set will apply.
- Rules without any categories are ignored when a `categories_enabled` set is present. This applies uniformly to input checks, output redaction, and action evaluation.
- Built-in PII rules are tagged with `{"pii", <pii_subtype>}`; enabling either `pii` or a specific subtype (e.g., `pii_email`) will activate those rules.

Runtime Overrides (Admin)
- Endpoints:
  - `GET /api/v1/moderation/settings` → runtime overrides + effective.
  - `PUT /api/v1/moderation/settings` → body `{pii_enabled?: bool, categories_enabled?: string[], persist?: bool}`.
- Persistence:
  - When `persist=true`, service writes to `runtime_overrides_file` and reloads policy.
  - Overrides load on startup and `POST /api/v1/moderation/reload`.

Admin API Endpoints
- `GET /api/v1/moderation/policy/effective?user_id=U` → effective policy snapshot.
- `POST /api/v1/moderation/reload` → reload config + overrides.
- Blocklist (managed):
  - `GET /api/v1/moderation/blocklist/managed` → `{version, items}` (sets `ETag`).
  - `POST /api/v1/moderation/blocklist/append` (requires `If-Match`) → append line.
  - `DELETE /api/v1/moderation/blocklist/{id}` (requires `If-Match`).
  - `PUT /api/v1/moderation/blocklist` (replace entire file).
  - `POST /api/v1/moderation/blocklist/lint` (dry-run validation) → validate one line or many without persisting.
    - Request: `{ line: string }` or `{ lines: string[] }`
    - Response: `{ items: [{ index, line, ok, pattern_type: 'literal'|'regex'|'comment'|'empty', action?, replacement?, categories?, error?, warning?, sample? }], valid_count, invalid_count }`
    - Notes: Use lint to pre-check regex safety (catastrophic patterns are rejected) and parse per-pattern actions (`block|redact|warn`) and `#categories` before appending or saving.
- Per-user Overrides:
  - `GET /api/v1/moderation/users` → list all.
  - `GET /api/v1/moderation/users/{user_id}` → get.
  - `PUT /api/v1/moderation/users/{user_id}` → upsert.
  - `DELETE /api/v1/moderation/users/{user_id}` → delete.
- Tester:
  - `POST /api/v1/moderation/test` → `{flagged, action, sample, redacted_text?, effective, category?}`.
  - Note: `sample` is a sanitized snippet (not the raw match or regex pattern). It redacts the matched portion using the effective redaction replacement to avoid exposing sensitive content.
  - Regex tester honors `/regex/flags` and category gating.

Web UI
- `/moderation`: Moderation Review queue. This is the reviewer workflow for sanitized items that were captured from moderation outcomes. Reviewers can filter/search/sort the queue, inspect sanitized context and policy snapshots, record decisions, undo recent eligible decisions, review decision history, and apply bulk decisions with partial-failure feedback.
- `/moderation/rules`: Content Rules configuration. This is the administrator workflow for runtime settings, managed blocklist, per-user overrides, and the tester sandbox.
- `/moderation-playground`: legacy redirect to `/moderation/rules`.

Moderation Review
- Purpose: queue moderation outcomes that need human review without exposing raw unsafe content.
- Capture gate:
  - `MODERATION_REVIEW_CAPTURE_ENABLED`: when truthy, supported moderation outcomes are captured into the review queue.
  - `MODERATION_REVIEW_DB_PATH`: optional SQLite path for review queue persistence. Defaults to `tldw_Server_API/Databases/moderation_review.db`.
- Review item data is intentionally sanitized:
  - `excerpt`, `context`, `effective_policy`, and `matches` are the only content-bearing fields surfaced to reviewers.
  - `safe_fields` tells the UI which fields are allowed to render.
  - Raw rule patterns and raw model/user text are not exposed through review item detail.
  - Redacted review items keep top-level metadata and audit records, but replace excerpt, context, and match samples with safe placeholders.
- Decision auditability:
  - Item detail includes sanitized decision history: actor id, action, resulting status, reason, timestamps, undo eligibility, undo expiry, and redaction state.
  - Undo tokens are returned only at decision time, are stored hashed, expire, are single-use, and fail if a later decision superseded the original decision.
  - `GET /api/v1/moderation/review/audit` lists sanitized audit events and supports filtering by item, decision, actor, action, date range, cursor, and limit.
- Review endpoints use moderation review permissions rather than the content-rules `SYSTEM_CONFIGURE` permission:
  - `MODERATION_REVIEW_READ`: list and inspect review items.
  - `MODERATION_REVIEW_DECIDE`: record and undo single-item decisions.
  - `MODERATION_REVIEW_BULK_DECIDE`: record bulk decisions.
  - `MODERATION_AUDIT_READ`: list sanitized review audit events.
- Known unsupported producer states:
  - Review capture currently covers moderation outcomes wired through the review capture helper. Additional producers should call `capture_moderation_review_item` with sanitized payloads rather than writing directly to the review database.
  - There is no standalone audit export endpoint yet. Use filtered audit listing until an export contract is designed.

Streaming Behavior
- Streaming SSE always ends with `data: [DONE]` on normal termination.
- When an output block occurs mid-stream, an SSE error payload is emitted, followed by `data: [DONE]`.

Metrics
- `chat_moderation_input_flag_total{user_id,action,category}`
- `chat_moderation_output_redact_total{user_id,category,streaming}`
- `chat_moderation_output_block_total{user_id,category,streaming}`
- `chat_moderation_stream_block_total{user_id,category}`
- Category label prefers a more specific subtype (e.g., `pii_email`) over generic `pii` when available.

Audit
- SECURITY_VIOLATION events on moderation actions with metadata: `{phase, action, pattern, streaming?}`.
- Blocks are recorded with `result=failure`, redactions with `result=success`.

Best Practices
- Prefer literals when possible; use bounded regexes.
- Avoid catastrophic patterns. The service rejects dangerous regex (nested quantifiers, excessive groups) and applies scan budgets.
- Use categories to enable optional rules (e.g., `pii`) selectively.

Testing
- Unit tests cover:
  - Input block 400, output redaction (non-stream), streaming redaction, streaming block with SSE error + `[DONE]`.
  - Categories gating for PII (`test_moderation_categories.py`).
- Run:
  - `python -m pytest -q tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
  - `python -m pytest -q tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

Notes
- Runtime overrides are non-destructive and can be removed by deleting keys or the overrides file.
