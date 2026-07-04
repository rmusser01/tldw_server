# Chat_Macros

`Chat_Macros` owns custom chat macro definitions, `/wrapup`, macro run
records, branch records, output profile resolution, Jobs dispatch, and the
small API/UI contract used by chat and workspace surfaces.

## Definition Format

Macro definitions are YAML files loaded with `yaml.safe_load` and validated by
Pydantic models. User macros are stored under each user's database base path as
`macros/<name>/MACRO.yaml` plus optional UTF-8 supporting files.

Command names in v1 are slash-compatible identifiers:

- Start with a lowercase letter.
- Continue with lowercase letters, numbers, or underscores.
- Maximum length is 64 characters.

Macro argument names follow the same lower/underscore style. The parser also
accepts hyphenated versions of underscore names, so `--output-profile` maps to
`output_profile`.

The v1 permission model is intentionally closed. Definitions with non-empty
`permissions.tool_calls` or `permissions.skills` are rejected during validation.

## Built-In `/wrapup`

`/wrapup` is bundled as a built-in macro. It runs multiple branch prompts over
the active chat/workspace context, then merges the retained branch outputs into
one final response.

Supported options:

- `--preset <name>`: choose a built-in question preset.
- `--question <text>`: add a custom question; may be repeated.
- `--output-profile <name>`: select a global output profile, falling back to
  `default` when missing.
- `--keep-forks`: retain scratch branches when the runner supports that mode.
- `--sync`: request synchronous behavior where the surface supports it.
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
- `POST /api/v1/chat/macros/run`: create and enqueue a macro run.
- `GET /api/v1/chat/macros/runs/{run_id}`: read run and branch detail.
- `POST /api/v1/chat/macros/runs/{run_id}/cancel`: request cancellation.

The WebUI exposes the minimal v1 manager at `/settings/chat-macros`. It lists
macros, toggles enabled state, clones `/wrapup`, edits settings JSON, validates
macro YAML, and renders workspace status cards plus lazy run detail.

## Execution And Jobs

Chat macro runs are persisted before branch execution. Background mode requires
the Jobs manager and the `chat_macros` Jobs worker; if the Jobs manager is not
available, `/wrapup` and direct `POST /run` fail closed instead of leaving an
unexecutable pending run.

The current Jobs worker builds a `ChatMacroExecutor` and supports cancellation,
branch/run status persistence, final output persistence, and idempotent
post-back. The conservative default branch runner returns failed branch records
until a real LLM/ACP runner is wired into the Jobs runtime.

## Security Notes

- Macro YAML and supporting files are size-limited and must be UTF-8.
- Supporting file names are constrained to simple path segments.
- Symlinked macro directories/files are rejected.
- Secret-like branch and run errors are redacted before API responses.
- Core slash command names are reserved; custom macros cannot shadow them.
