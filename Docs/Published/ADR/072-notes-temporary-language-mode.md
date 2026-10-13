# ADR-072: Temporary language display on incapable direct Notes servers

**Status:** Accepted
**Date:** 2026-10-11
**Decision owner:** TASK-13595 controller, Ruling48
**Related task:** TASK-13595
**Related spec:** Chatbook Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md
**Related plan:** Chatbook Docs/superpowers/plans/2026-09-27-structured-document-editing.md
**Backfilled from:** N/A — controller decision recorded before implementation
**Extends:** [ADR-070](070-portable-note-language-and-negotiated-structured-editing.md)

## Decision

The existing `useNotesEditorState` canonical owner may carry one ephemeral display-language override for the current note/session. This override is permitted only after an authenticated, owner-scoped capability response proves the server cannot persist language and the successful canonical load proves the language field is absent. Missing capability fields, transport/auth errors, or an ambiguous recovery snapshot do not supply that proof.

Canonical `content_language`, its presence, fresh-default provenance, draft/recovery payload and protected save acknowledgements remain independent of the display override. Existing known/null/future canonical modes and explicit portable recovery intent stay protected. An ordinary absent-language body/title save continues to omit metadata; the override is never sent, recovered, synchronized or described as portable. No second store, provider or automatic portable promotion is added.

The override survives ordinary body/title saves and benign same-note reload while canonical absence and owner authority still hold. Selection, account/auth/connection changes and newly present canonical metadata retire it permanently. If a fresh note uses the already qualified untouched-default proof, it remains temporary only through that same origin lineage; historical missing/null intent never inherits that exception. Effective mode changes source presentation and checking without rewriting content.

## Context

The approved programme explicitly allows a clearly labeled temporary language selection for a directly server-backed note on an incapable server. F5 protects portable intent and recognizes omission separately from explicit null. Reusing the persisted mode as a temporary view would incorrectly add a capability veto to ordinary old-server saves or silently downgrade acknowledgements.

## Alternatives considered

- Persisting the display selection as canonical intent makes safe ordinary body saves fail or falsely claims portable persistence.
- Downgrading all mode acknowledgements discards explicit null/future/protected intent.
- Treating absent capability fields as an incapable-server proof guesses across unknown/auth/transport states.
- A new local metadata store duplicates the existing draft owner and implies unapproved durability.

## Consequences and qualification

The selector reports temporary versus portable/unavailable mode honestly. Canonical and effective language have distinct semantics in the existing owner interface. Actual owner tests must cover absent/known/null/future/recovered intent, late acknowledgements, same-note reload, navigation and authority restoration. The original default Markdown/WYSIWYG regression and metadata dirty/recovery contract remain governed by ADR-070. Physical IME and release profiles remain I5 gates.

## Renumbering provenance

This accepted decision was first recorded as ADR-067 in the reviewed structured-document branch and is published as ADR-072 to keep the programme ADR block distinct from current dev allocations. Its decision, original approval and implementation evidence are unchanged; original reviewed commits retain the historical identifier.
