# Skills

Skills manages user-defined skill bundles stored as `SKILL.md` plus optional supporting files. The package parses skill metadata, synchronizes the per-user skill filesystem with ChaChaNotes registry rows, exposes skill context to chat/persona flows, and executes skill prompts with constrained tool access.

## Start Here

- `skills_service.py` is the primary CRUD, import/export, filesystem, and registry synchronization service.
- `skill_parser.py` parses `SKILL.md` content and front matter.
- `skill_executor.py` resolves arguments, allowed tools, and execution payloads for chat-time skill use.
- Related API surface: `app/api/v1/endpoints/skills.py`.
- Related schemas: `app/api/v1/schemas/skills_schemas.py`.
- Related tests: `tests/Skills/`.

## Responsibilities

- Create, update, rename, delete, import, and export per-user skill directories under the user's database storage area.
- Validate skill names, supporting filenames, file counts, and total supporting-file byte limits.
- Keep skill registry metadata in ChaChaNotes in sync with the filesystem.
- Parse skill descriptions, allowed tools, execution mode, and prompt body from `SKILL.md`.
- Build skill context payloads for chat/persona consumers.
- Substitute skill arguments and filter available tools before execution.

## Module Map

- `skills_service.py` - filesystem-backed skill management plus registry persistence.
- `skill_parser.py` - parser for skill markdown and metadata.
- `skill_executor.py` - argument substitution, tool filtering, and inline/fork execution helpers.
- `context_integration.py` - context bridge used by chat and persona routes.
- `exceptions.py` - module-specific exception types.
- `builtin/` - package namespace for built-in skill assets.

## How It Connects

- `app/api/v1/endpoints/skills.py` exposes the REST API for listing, reading, writing, importing, exporting, and executing skills.
- `app/core/Chat/command_router.py` can execute skills from chat commands.
- `app/api/v1/endpoints/chat.py` and `app/api/v1/endpoints/persona.py` use skill context integration when preparing chat/persona context.
- ChaChaNotes stores registry metadata while the canonical editable content lives in per-user files.

## Extension Points

- For new skill metadata fields, update `skill_parser.py`, `skills_schemas.py`, and registry serialization in `skills_service.py`.
- For new execution behavior, inspect `skill_executor.py` and the chat command router before changing endpoint code.
- For import/export changes, start with `skills_service.py` and add tests under `tests/Skills/unit/`.
- For chat context changes, update `context_integration.py` and the relevant chat/persona tests.

## Testing

- `tests/Skills/unit/`
- `tests/Skills/integration/`
- Chat command and context behavior is also covered by chat/persona tests that import `app/core/Skills/context_integration.py`.

## Gotchas

- Skill files are user-editable, so path traversal checks, filename validation, and file-size limits are part of the security boundary.
- Tool access must be filtered through the skill's allowed-tool configuration before execution.
- Registry rows and filesystem state can drift; `skills_service.py` includes synchronization logic that callers should reuse instead of reimplementing.
