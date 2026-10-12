# ADR-073: Exact source for secondary Notes capture

**Status:** Accepted
**Date:** 2026-10-11
**Decision owner:** TASK-13595 controller, Ruling50
**Related task:** TASK-13595
**Related spec:** Chatbook Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md
**Related plan:** Chatbook Docs/superpowers/plans/2026-09-27-structured-document-editing.md
**Backfilled from:** N/A — controller decision recorded before implementation
**Extends:** [ADR-070](070-portable-note-language-and-negotiated-structured-editing.md)

## Decision

The existing secondary Notes capture action saves the exact selected/edited source. The confirmed background `saveToNotesMenuId` branch and the sidepanel route use trimming only for emptiness validation. The route submits the existing canonical title and `captured` tag without inferred language, null metadata or an appended Markdown source line.

The same action is labeled **Save source** and explains before saving that only text is saved and the displayed page URL is not included. The page URL remains visible while the draft is open. Existing note bodies, including their previously stored inline source attribution, are unchanged. No new provenance field, storage, option, action or handoff subsystem is introduced.

## Context and evidence

The sole production `buildCapturedNoteContent` caller trimmed the draft and appended `Source: ...`. A confirmed upstream context-menu branch trimmed the selection before forwarding it. These transformations can alter JSON/YAML/JSONL/text and also alter literal Markdown source.

An authenticated ordinary API POST201/GET200 against task-owned SQLite preserved exact CRLF/Unicode/source punctuation but ignored nested `metadata.source_url`. Actual NoteCreate/NoteResponse define no general metadata field. That is evidence of absence, not permission to introduce a schema. Canonical keywords cap individual tags at100 characters and deduplicate case-insensitively, so opaque URL identity cannot safely be encoded as a tag.

## Alternatives considered

- Keep the Markdown attribution append: alters structured and literal source without an explicit conversion action.
- Move URL into nested metadata: the actual API silently ignores it.
- Encode URL as a keyword: inherits incompatible length, case identity, visibility and Sync semantics.
- Add a provenance schema/store or new action/option: unnecessary expansion beyond source-safe secondary capture.

## Consequences and qualification

New captures lose persisted inline page attribution deliberately and visibly. Exact selected text and existing title/tag/error/authority behavior remain. Actual mounted route controls must verify raw JSON, YAML, JSONL, Markdown and text; the real create/read/export path must preserve the submitted source. The modal keeps canonical source when native textarea representation is lossy. Other capture, transfer, narration and existing Notes remain outside this correction.

## Renumbering provenance

This accepted decision was first recorded as ADR-068 in the reviewed structured-document branch and is published as ADR-073 to keep the programme ADR block distinct from current dev allocations. Its decision, original approval and implementation evidence are unchanged; original reviewed commits retain the historical identifier.
