# Chat_Macros

`Chat_Macros` owns custom chat macro definitions, `/wrapup`, macro run
records, branch records, output profile resolution, Jobs dispatch, and the
API/UI contract used by chat and workspace surfaces.

## Definition Format And Ownership

Macro definitions are YAML files loaded with `yaml.safe_load` and validated by
Pydantic models. User macros are stored beneath the owning user's database base
path at `macros/<name>/MACRO.yaml`. The directory and definition belong to that
user; a user cannot read, update, or delete another user's macros.

The resource name is immutable after creation. The API route name, storage
directory, and YAML `name` must agree. Create and update reject identity
mismatches, while clone generates matching resource and YAML identities.
Standalone validation validates YAML only because it has no resource identity to
compare. Commands remain separately configurable subject to command validation.

Command names in v1 are slash-compatible identifiers:

- Start with a lowercase letter.
- Continue with lowercase letters, numbers, or underscores.
- Maximum length is 64 characters.

Macro argument names follow the same lower/underscore style. The parser also
accepts hyphenated versions of underscore names, so `--output-profile` maps to
`output_profile`.

The v1 permission model is intentionally closed. Definitions with non-empty
`permissions.tool_calls` or `permissions.skills` are rejected during server-side
validation. Validation is authoritative: client-side guided checks improve
authoring feedback but never bypass the server.

## Authoring In The WebUI

`/settings/chat-macros` provides the v1.1 manager. It keeps the catalog and
editor visible together on wide viewports, then stacks them on narrow screens.
The manager supports:

- Creating a blank user macro, editing it, and deleting it with an accessible
  confirmation that defaults focus to cancel.
- Selecting any catalog entry to inspect its command, source, enabled state,
  and catalog validation status. `GET /api/v1/chat/macros` returns
  `validation_status` and `validation_error` for every catalog summary; entries
  that successfully reach this catalog are explicitly reported as `valid` with
  no validation error.
- Enabling or disabling built-in and user macros without changing the YAML.
- Cloning the currently selected built-in into a user macro. Built-ins remain
  read-only: they expose disable and clone actions, not edit or delete actions.
- Refreshing the catalog after create, update, delete, clone, or enable changes
  while retaining the matching selection whenever it still exists.

Import and export are YAML-only. An import opens a blank editor draft and does
not persist until the user saves after validation. Imports accept `.yaml` or
`.yml` source files only and are limited to 500,000 bytes. Exports include the
visible draft, including unsaved edits. Untouched definitions retain the exact
canonical YAML returned by the server.

Guided mode intentionally supports a narrow, inspectable topology: one through
six ordered `branch_prompt` steps, followed by one `merge` step and one
`post_result` step, with the default v1 context, retry, permission, and
background-execution settings. YAML outside that topology remains editable in
Source mode. Switching to Guided is refused when it would discard unsupported
fields or topology.

## Output Profiles

Macro settings contain named output profiles. The output-profile editor saves
only profiles through an atomic backend update, preserving current macro toggles
and unrelated settings. Unsaved profile edits survive settings refreshes.

Each profile selects one result format:

- `structured_sections`: return the configured ordered sections.
- `single_response`: return one consolidated response.

Profiles can include branch outputs and define custom `section_titles` for
individual section keys. Section order is significant. A profile has one to ten
sections; section keys use the same lowercase identifier rules as profile names,
and custom titles are trimmed, nonblank, and bounded to 128 characters. Existing
profile names accepted by the backend remain editable. The default profile is
always retained.

## Built-In `/wrapup`

`/wrapup` is bundled as a built-in macro. It runs multiple branch prompts over
the active chat/workspace context, then merges retained branch outputs into one
final response.

Supported options:

- `--preset <name>`: choose a built-in question preset.
- `--question <text>`: add a custom question; may be repeated.
- `--output-profile <name>`: select a global output profile, falling back to
  `default` when missing.
- `--keep-forks`: retain scratch branches when the runner supports that mode.
- `--include-branches`: include branch outputs in the final response when the
  selected output profile allows it.

Unknown, duplicate non-repeated, malformed, or over-limit arguments fail before
LLM dispatch and return a chat-visible validation error.

## API And Settings

The REST API is exposed under `/api/v1/chat/macros`:

- `GET /api/v1/chat/macros`: list built-in and user macros.
- `GET /api/v1/chat/macros/{name}`: get a macro definition.
- `POST /api/v1/chat/macros`: create a user macro.
- `PUT /api/v1/chat/macros/{name}`: replace a user macro YAML or toggle
  `enabled` for built-in/user macros.
- `DELETE /api/v1/chat/macros/{name}`: delete a user macro.
- `POST /api/v1/chat/macros/{name}/clone`: clone a built-in macro to a user
  macro.
- `POST /api/v1/chat/macros/validate`: validate macro YAML without saving it.
- `GET|PUT /api/v1/chat/macros/settings`: read or replace macro settings,
  including global output profiles and disabled built-ins.
- `PUT /api/v1/chat/macros/settings/output-profiles`: atomically replace only
  output profiles, preserving other current settings.
- `POST /api/v1/chat/macros/run`: create and enqueue a macro run.
- `GET /api/v1/chat/macros/runs/{run_id}`: read run and branch detail.
- `POST /api/v1/chat/macros/runs/{run_id}/cancel`: request cancellation.

Previously stored empty section lists fall back to default sections on read,
and whitespace-only headings fall back to generated headings. Stored heading
line breaks and control characters become spaces. This keeps legacy settings
editable; new writes require nonblank, single-line, control-free headings.

## Execution And Jobs

Chat macro runs are persisted before branch execution. V1 supports background
execution only; foreground and synchronous modes are deferred and rejected
instead of being silently enqueued. Background mode requires the Jobs manager
and the `chat_macros` Jobs worker; if the Jobs manager is not available,
`/wrapup` and direct `POST /run` fail closed instead of leaving an unexecutable
pending run.

The current Jobs worker builds a `ChatMacroExecutor`, runs chat-native branches
through the canonical async LLM provider service, and supports cancellation,
branch/run status persistence, final output persistence, and idempotent
post-back. ACP-specific fork execution remains capability-gated and falls back
to chat-native execution when the macro allows it. ACP fork-retention controls
are not yet exposed in the v1.1 manager.

## Security And Path Safety

- Macro YAML is size-limited, must be UTF-8, and receives server-side schema,
  permission, command, and identity validation before it is stored or run.
- Supporting file names are constrained to simple path segments; symlinked macro
  directories and files are rejected to keep user-owned macro storage path-safe.
- Secret-like branch and run errors are redacted before API responses.
- Core slash command names are reserved; custom macros cannot shadow them.

## Deferred Scope

The v1.1 authoring workflow deliberately defers supporting-file import/export
bundles, ACP fork-retention UI, foreground execution, and additional built-in
presets. These gaps do not relax validation, ownership, path-safety, or
background-job requirements for the functionality that is available now.
