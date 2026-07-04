# API Module Documentation Design

## Goal

Make the API documentation easier for users to scan by explaining what each API module can do, what kinds of workflows it supports, and where to find the detailed reference material.

## Reviewed Scope

The API surface is broader than the current `API_Tags_Index.md` and broader than the curated `OPENAPI_TAGS` list in `tldw_Server_API/app/main.py`. Router groups expose many stable, experimental, admin, and support tags. The documentation pass should therefore cover modules at the OpenAPI tag level, not every endpoint.

The work has two deliverables:

1. Expand `Docs/API-related/API_Tags_Index.md` into a grouped capability guide.
2. Align OpenAPI tag metadata and ReDoc grouping in `tldw_Server_API/app/main.py` with the same user-facing organization.

`Docs/Published` is generated output and should not be edited manually in this task. The publishing process is responsible for regenerating the published mirror from the source guide.

## User-Facing Structure

The capability guide should group modules by what a user is trying to do:

- Start, authenticate, and configure the server.
- Ingest, process, and organize media.
- Transcribe, synthesize, and interact with audio.
- Chat with models, characters, personas, and tools.
- Search, retrieve, embed, and evaluate knowledge.
- Manage notes, prompts, study material, and generated artifacts.
- Automate workflows, jobs, schedules, and integrations.
- Administer users, organizations, usage, storage, security, and operations.
- Use experimental surfaces such as ACP, VN, sandbox, companion, and guardrails.

Each row should include the tag/module name, plain-language capabilities, common use cases, and the most relevant existing documentation link when one exists. Tags without dedicated docs should still explain what the module is for instead of only saying "Coming soon."

## OpenAPI Metadata Design

`OPENAPI_TAGS` should remain concise metadata for API browsers, not become the full guide. Add or improve tag entries for important router tags that currently appear only as auto-filled names. Descriptions should answer "what can I do here?" in one sentence.

The `x-tagGroups` list should match the guide's high-level categories so ReDoc users can browse by capability. Include the most relevant stable and common experimental tags. Tags not listed in groups still remain visible through FastAPI/OpenAPI because `_ensure_openapi_operation_tags_declared` adds missing tags.

## Risk Controls

- Do not change route registration, prefixes, security, schemas, dependencies, or endpoint behavior.
- Avoid documenting every operation inline in `main.py`; detailed examples belong in markdown docs.
- Label admin-only and experimental surfaces clearly so users do not mistake support APIs for primary workflows.
- Preserve existing docs links where they are valid, and prefer existing detailed docs over creating new deep references in this pass.
- Keep the guide reviewable as one documentation unit. If a module needs a full how-to, leave a pointer or future-doc note rather than expanding this task.

## Verification

Verification should include:

- OpenAPI schema generation or import smoke check after `main.py` changes.
- Markdown sanity check for the edited source API tag index file.
- Link existence check for local markdown links added or retained in the edited guide.
- No branch diff for `Docs/Published/API-related/API_Tags_Index.md`, because Published docs are generated.
- Bandit against `tldw_Server_API/app/main.py` if that Python file is changed.

## Non-Goals

- No endpoint behavior changes.
- No full rewrite of every existing API guide.
- No new docs publishing pipeline.
- No new generated OpenAPI artifact unless the repository already expects one.
