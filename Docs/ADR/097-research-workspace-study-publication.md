# ADR-097 · Publish Research Studio Study outputs atomically

**Status:** Proposed
**Date:** 2026-11-24
**Backfilled from:** New decision
**Decision owner:** Research Workspace workstream
**Related task:** TASK-15677
**Related design:** [Research Workspace Study publication](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/research-workspace-study-publication.md)

## Decision

Commit each authenticated Research Studio deck or quiz, its entries, and a
workspace-scoped replay receipt in one native ChaCha transaction. Study
continues to own the deck or quiz. The receipt holds its native ID and bounded
generation provenance.

## Context

The existing Study endpoints create a parent and its entries in separate
requests. If a client loses the response or fails after creating some entries,
it cannot establish what committed. Chatbook needs to publish generated items
into Studio and reopen their exact native Study records.

## Alternatives considered

- Sequential Study requests would expose partially populated decks and quizzes
  and assign another ID on retry.
- Workspace artifacts would create a second representation of study content and
  require extra synchronization when the canonical Study owner changes.

## Consequences

Workspace and authenticated owner qualify every receipt lookup. Identical
requests recover their owner after a response is lost. Reusing an operation
with a different request or replaying against a missing or moved native owner
fails. SQLite and PostgreSQL each receive a schema migration. Existing Study
edit and delete operations retain ownership of the canonical record.
