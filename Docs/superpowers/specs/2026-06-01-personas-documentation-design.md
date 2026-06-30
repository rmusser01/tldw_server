# Personas Documentation Design

## Goal

Create source documentation for the Personas feature and refresh the Persona core module README so users and contributors can understand the current implementation without relying on generated published docs or older design notes.

## Scope

This work updates source documentation only:

- `Docs/User_Guides/Server/Personas_User_Guide.md`
- `tldw_Server_API/app/core/Persona/README.md`

`Docs/Published/` is intentionally out of scope because it is generated.

## User Guide Design

The user guide explains Personas as persistent assistant identities that can carry instructions, state docs, exemplars, scope rules, policy rules, live sessions, voice commands, and visual packs across user-owned workflows. It distinguishes Personas from Character Chat: characters are roleplay/chat cards, while Personas are broader assistant profiles for chat, live sessions, tools, and workflow context.

The guide includes prerequisites, authentication notes, a curl quickstart, WebSocket usage notes, common concepts, privacy and safety boundaries, and troubleshooting. Examples use `/api/v1/persona` routes and avoid claiming generated docs output.

## Module README Design

The core README becomes a developer-oriented module guide. It maps responsibilities, file layout, request/session lifecycle, persistence in `ChaChaNotes_DB`, runtime concepts, visual-pack boundaries, security constraints, extension points, and targeted tests.

The README keeps operational guidance concise and code-facing. It points to the user guide for workflow examples and to existing visual-pack docs for the deeper visual asset contract.

## Verification

Verification is documentation-oriented:

- Confirm the two intended source docs exist.
- Confirm no `Docs/Published/` files were modified.
- Run markdown/link sanity checks available without network.
- Run Bandit only if Python code changes occur; otherwise record the non-code skip in `TASK-586`.
