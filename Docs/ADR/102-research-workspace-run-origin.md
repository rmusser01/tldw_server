# ADR-102 · Retain the origin of a Workspace Research run

**Status:** Proposed
**Date:** 2026-11-24
**Decision owner:** Research Workspace workstream
**Related task:** TASK-15638
**Related design:** [Research Workspace run origin](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/research-workspace-run-origin.md)

## Decision

The authenticated Workspace launch route validates the Workspace and its captured
source associations. The native Research store commits the run, its first
phase intent, and an owner-scoped launch receipt in one transaction. The
receipt retains a bounded origin and the exact native run ID. Run detail
projects that origin to the authenticated owner.

## Context

Workspace and Runs have different lifecycles. WebUI route state preserves a
workspace ID while navigating. Chatbook also needs to recover the exact
launch after process restart or a lost response and verify the destination
before a user imports a Report.

## Alternatives considered

- Route state can be lost and does not establish that the backend accepted the
  origin.
- The Chat handoff has a different purpose. It delivers a completion message
  into a conversation and has its own access and delivery lifecycle.
- Workspace-owned Runs would bypass the native run, checkpoint, and Jobs
  contracts.

## Consequences

A retry recovers the same native run and dispatch intent. Reusing a launch
identity with a different request fails. Deleting the run removes the return
context and retains a thin receipt that prevents creation on replay.
Workspace context holds references and the normalized question. Device
profile identifiers, source bodies, and Report text stay out of the receipt.
