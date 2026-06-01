# Personas Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create source documentation for the Personas feature and refresh the Persona core module README.

**Architecture:** Keep user-facing workflow documentation in `Docs/User_Guides/Server/` and developer/module documentation beside the code in `tldw_Server_API/app/core/Persona/README.md`. Do not edit `Docs/Published/` because the published tree is generated.

**Tech Stack:** Markdown documentation, FastAPI route references, Persona core Python module map, Backlog.md CLI tracking.

---

## File Structure

- Create: `Docs/User_Guides/Server/Personas_User_Guide.md`
  - Source user guide for Personas workflows, concepts, API examples, privacy, safety, and troubleshooting.
- Modify: `tldw_Server_API/app/core/Persona/README.md`
  - Developer/module guide for core Persona responsibilities, module map, lifecycle, persistence, extension points, and tests.
- Update: `backlog/tasks/task-586 - Document-Personas-feature-and-core-module.md`
  - Task notes, verification results, touched files, and final summary through the Backlog CLI.

## Task 1: Write Personas User Guide

**Files:**
- Create: `Docs/User_Guides/Server/Personas_User_Guide.md`

- [x] **Step 1: Create the source user guide**

Add a Markdown guide with these sections:

```markdown
# Personas User Guide

Last Updated: 2026-06-01

## Overview
## Personas vs Characters
## Prerequisites
## Quickstart
## Core Concepts
## Live Sessions and WebSocket Stream
## State, Memory, and Exemplars
## Scope and Policy Rules
## Voice Commands and Connections
## Visual Packs
## Privacy and Safety
## Common Errors
## Related Docs
```

- [x] **Step 2: Check route names against current code**

Run:

```bash
rg -n '@router\.(get|post|put|patch|delete|websocket)' tldw_Server_API/app/api/v1/endpoints/persona.py
```

Expected: output includes `/profiles`, `/catalog`, `/session`, `/sessions`, `/live/sessions`, and `/stream`.

## Task 2: Refresh Core Persona README

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/README.md`

- [x] **Step 1: Replace the compact status note with a developer module guide**

Use these sections:

```markdown
# Persona Module

## Purpose
## Responsibilities
## Module Map
## Runtime Lifecycle
## Persistence Model
## API Touch Points
## Runtime Concepts
## Visual Packs and Buddy Runtime
## Security and Privacy Boundaries
## Configuration
## Testing
## Extension Guidance
## Common Pitfalls
```

- [x] **Step 2: Keep links source-oriented**

Reference `Docs/User_Guides/Server/Personas_User_Guide.md` and `Docs/Code_Documentation/Persona_Visual_Packs.md`. Do not reference `Docs/Published/` as an editable source.

## Task 3: Verify Documentation Scope

**Files:**
- Inspect: `Docs/User_Guides/Server/Personas_User_Guide.md`
- Inspect: `tldw_Server_API/app/core/Persona/README.md`
- Inspect: `git diff --name-only`

- [x] **Step 1: Confirm no generated published files were edited**

Run:

```bash
git diff --name-only | rg '^Docs/Published/' || true
```

Expected: no output.

- [x] **Step 2: Run whitespace check**

Run:

```bash
git diff --check -- Docs/User_Guides/Server/Personas_User_Guide.md tldw_Server_API/app/core/Persona/README.md
```

Expected: no output and exit code 0.

- [x] **Step 3: Record Bandit skip**

Because this plan touches Markdown only, record in Backlog that Bandit was skipped as non-code.

## Task 4: Backlog Closeout

**Files:**
- Update: `backlog/tasks/task-586 - Document-Personas-feature-and-core-module.md`

- [x] **Step 1: Add final notes**

Record touched files, verification commands, and the Bandit non-code skip.

- [x] **Step 2: Mark task done**

Run:

```bash
backlog task edit 586 --status Done --plain
```

Expected: task status is Done.
